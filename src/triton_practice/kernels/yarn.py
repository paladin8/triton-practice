"""
Kernels for computing RoPE with context extension via YaRN (Yet another RoPE extensioN method).
See paper for details: https://arxiv.org/pdf/2309.00071

Look at rope.py to see a description of the base RoPE algorithm.

Attention parameters:
B = batch size
S = sequence length
H = number of attention heads
D = hidden dimension per head

RoPE parameters:
b = rotation base

YaRN parameters:
L = original context length
s = scaling factor
alpha = "far" threshold
beta  = "close" threshold

YaRN modifies base RoPE as follows.
  1. The base RoPE angle in the sequence and dimensions {2*d, 2*d+1} is:
        theta_d = b ** (-2 * d / D)
  2. Suppose we want to increase the context length from its original trained value by a factor of s. Then we can
     imagine an interpolated rotation angle of:
        phi_d = theta_d / s
     which simply slows down the rotations by a factor of s so that the extended context length covers the same range
     of angles as the original context length.
  3. It turns out that for low values of d, you want to use an angle closer to theta_d, while for high values of d,
     you want to use an angle closer to phi_d. The theory is that "close" tokens should influence in the same way even
     after extending the context length, while "far" tokens should have their influence normalized relative to the
     context length.

     Define the quantity r(d) by: r(d) = (L / (2 * pi)) * theta_d
     The final rotation angle f(d) is determined piecewise by:
        r(d) > beta            =>  f(d) = theta_d
        r(d) < alpha           =>  f(d) = phi_d
        alpha <= r(d) <= beta  =>  f(d) = (theta_d * (r(d) - alpha) + phi_d * (beta - r(d)) / (beta - alpha)
     i.e. theta_d for high r(d), phi_d for low r(d), and a linear interpolation of the two in between.
  4. Finally, we apply a "length scaling" trick to balance the fact that the attention dot products are computed over
     longer sequences. We do this by multiplying the rotated queries and keys by a factor of 0.1 * log(s) + 1.

Although the paper defines f(d) as above, all implementations I could find use a slightly different formula. According
to those implementations, step 3 should instead be as follows.
   3. Let r' be the inverse of r and define: alpha' = ceil(r'(alpha)), beta' = floor(r'(beta)). Then f(d) is:
         d < beta'             =>  f(d) = theta_d
         d > alpha'            =>  f(d) = phi_d
         alpha' <= d <= beta'  =>  f(d) = (theta_d * (alpha' - d) + phi_d * (d - beta')) / (alpha' - beta')
      which is a similar idea but interpolated differently (because r/r' are not linear).
"""

import math
from typing import Literal, cast

import torch
import triton
import triton.language as tl
import triton.testing
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import YaRNScalingRotaryEmbedding

from triton_practice.utils.device import DEVICE


def vllm_yarn(
    qk: torch.Tensor,  # [B, S, 2, H, D] tensor containing queries and keys
    b: float,  # rotation base
    L: int,  # original context length
    s: float,  # scaling factor
    alpha: int,  # "far" threshold
    beta: int,   # "close" threshold
) -> torch.Tensor:
    B, S, _, _, D = qk.shape
    assert D % 2 == 0, "D must be even"
    assert beta >= alpha, "Close threshold must be at least as high as far threshold"

    yarn_embedding = YaRNScalingRotaryEmbedding(
        head_size=D,
        rotary_dim=D,
        max_position_embeddings=L,
        base=b,
        is_neox_style=False,  # interleaved rotations
        scaling_factor=s,
        dtype=qk.dtype,
        beta_slow=alpha,
        beta_fast=beta,
    )
    yarn_embedding.forward_cuda(
        positions=torch.arange(0, S).repeat(B),
        query=qk[:, :, 0, :, :].view(B * S, -1, D),
        key=qk[:, :, 1, :, :].view(B * S, -1, D),
    )

    return qk


def _invert_r(alpha: float, beta: float, D: int, b: float, L: int) -> tuple[float, float]:
    alpha_p = min(D-1, math.ceil((D / 2) * math.log(L / (2 * math.pi * alpha)) / math.log(b)))
    beta_p = max(0, math.floor((D / 2) * math.log(L / (2 * math.pi * beta)) / math.log(b)))

    # This version is what I would do to bound alpha_p/beta_p relative to each other.
    # if alpha <= beta_p:
    #     alpha_p = beta_p + 0.001

    # This version matches the vLLM implementation. In practice, this is an edge case that probably doesn't affect
    # performance given a real LLM's model parameters.
    if alpha_p == beta_p:
        alpha_p += 0.001

    return alpha_p, beta_p


def torch_yarn(
    qk: torch.Tensor,  # [B, S, 2, H, D] tensor containing queries and keys
    b: float,  # rotation base
    L: int,  # original context length
    s: float,  # scaling factor
    alpha: int,  # "far" threshold
    beta: int,   # "close" threshold
) -> torch.Tensor:
    _, S, _, _, D = qk.shape
    assert D % 2 == 0, "D must be even"
    assert beta >= alpha, "Close threshold must be at least as high as far threshold"

    # Compute the theta_d and phi_d angles.
    dimensions = torch.arange(D//2, device=DEVICE, dtype=torch.float32)  # [D/2] indices
    thetas = b ** (-2 * dimensions / D)  # [D/2] angles
    phis = thetas / s  # [D/2] angles

    # This version is my interpretation of the YaRN paper.
    # Compute r(d) and use a clamped linear interpolation to compute f(d).
    # r = (L / (2 * math.pi)) * theta  # [D/2]
    # interp = torch.clamp((r - alpha) / (beta - alpha), min=0, max=1)  # [D/2] interpolation factor
    # f = interp * theta + (1 - interp) * phi  # [D/2] angles

    # This version matches the implementations of YaRN I could find (including the official repository).
    # First compute alpha' and beta' by inverting r, bounding them in an appropriate range. Then interpolate f between
    # theta and phi per the equation above.
    alpha_p, beta_p = _invert_r(alpha=alpha, beta=beta, D=D, b=b, L=L)
    interp = torch.clamp((alpha_p - dimensions) / (alpha_p - beta_p), min=0, max=1)  # [D/2] interpolation factor
    f = interp * thetas + (1 - interp) * phis  # [D/2] angles

    # Compute the rotation angles.
    position_ids = torch.arange(S, device=DEVICE, dtype=torch.float32)[:, None]  # [S, 1] indices
    rotation_angles = torch.matmul(position_ids, f[None, :])  # [S, D/2] angles

    # Compute the rotation sin/cos.
    sin = torch.sin(rotation_angles).to(dtype=qk.dtype)  # [S, D/2]
    cos = torch.cos(rotation_angles).to(dtype=qk.dtype)  # [S, D/2]

    # Expand the shapes of sin/cos to enable broadcasting.
    sin = sin[None, :, None, None, :]  # [1, S, 1, 1, D/2]
    cos = cos[None, :, None, None, :]  # [1, S, 1, 1, D/2]

    # Split qk tensor by the last dimension (hidden vector). qk_0 contains the even indices, and qk_1 the odd ones.
    qk_0 = qk[..., 0::2]  # [B, S, 2, H, D/2]
    qk_1 = qk[..., 1::2]  # [B, S, 2, H, D/2]

    # Apply rotations using element-wise multiplications between the qk splits and sin/cos. The sin/cos tensors are
    # automatically broadcast along the incomplete dimensions, so you can think of this as applying the [S, D/2]
    # versions of sin/cos to each batch/qk/head slice of the tensor.
    qk_rot_0 = qk_0 * cos - qk_1 * sin  # [B, S, 2, H, D/2]
    qk_rot_1 = qk_0 * sin + qk_1 * cos  # [B, S, 2, H, D/2]

    # Apply length scaling and store results back into the qk tensor.
    length_scale = 0.1 * math.log(s) + 1
    qk[..., 0::2] = qk_rot_0 * length_scale
    qk[..., 1::2] = qk_rot_1 * length_scale

    return qk


torch_compile_yarn = torch.compile(torch_yarn)


# Each run of this kernel is responsible for a stride of rows in the output.
@triton.autotune(
    configs=[
        # Multiple configurations for triton to choose from. Triton will benchmark these and pick the best one.
        # Stages (typically 2-4): more stages allow for more overlapping of memory traffic with compute, but at the cost
        # of more SHMEM/register usage.
        # Warps (typically 2, 4, 8): more warps can increase occupancy, but at the cost of more SHMEM/register usage.
        triton.Config({"S_STRIDE": 64}, num_stages=2, num_warps=4),
        triton.Config({"S_STRIDE": 64}, num_stages=2, num_warps=8),
        triton.Config({"S_STRIDE": 256}, num_stages=2, num_warps=4),
        triton.Config({"S_STRIDE": 256}, num_stages=2, num_warps=8),
    ],
    key=["S"],
)
@triton.jit
def _triton_rope_sin_cos_kernel(
    sin_ptr: tl.pointer_type,  # [S, D/2] output sin tensor
    cos_ptr: tl.pointer_type,  # [S, D/2] output cos tensor
    S: int,  # sequence length
    D_half: tl.constexpr,  # hidden dimension per head (half)
    b: float,  # rotation base
    L: int,  # original context length
    s: float,  # scaling factor
    alpha_p: int,  # inverted "far" threshold
    beta_p: int,   # inverted "close" threshold
    stride_sins: int,  # stride of sin tensor along the sequence length
    stride_sind: int,  # stride of sin tensor along the hidden dimension
    stride_coss: int,  # stride of cos tensor along the sequence length
    stride_cosd: int,  # stride of cos tensor along the hidden dimension
    S_STRIDE: tl.constexpr,  # number of kernel instances along the sequence length
) -> None:
    # Identify which block in the grid this kernel instance is responsible for.
    pid_s = tl.program_id(axis=0)  # program ID in the first dimension

    # Iterate over rows assigned to this program ID. We stride the rows instead of processing contiguous groups of rows
    # because that leads to better memory access patterns across the kernel instances.
    for m in range(pid_s, S, S_STRIDE):
        # Note that we omit masking here because we assume D is always a power of 2.
        offsets_d = tl.arange(0, D_half)  # offsets along the hidden dimension
        thetas = tl.exp(-tl.log(b) * offsets_d / D_half)  # [D/2] angles
        phis = thetas / s  # [D/2] angles

        # Interpolate f between theta and phi per the equation above and multiply by m to get the rotation angles.
        interp = tl.clamp((alpha_p - offsets_d) / (alpha_p - beta_p), min=0, max=1)  # [D/2] interpolation factor
        f = interp * thetas + (1 - interp) * phis  # [D/2] angles
        rotation_angles = m * f  # [D/2] angles

        # Compute sin and cos tiles.
        sin_tile = tl.sin(rotation_angles)  # [D/2] tile
        cos_tile = tl.cos(rotation_angles)  # [D/2] tile

        # Store the tiles into the sin and cos tensors at the right offsets.
        sin_offsets = m * stride_sins + offsets_d * stride_sind  # [D/2] offsets
        cos_offsets = m * stride_coss + offsets_d * stride_cosd  # [D/2] offsets
        tl.store(sin_ptr + sin_offsets, sin_tile)
        tl.store(cos_ptr + cos_offsets, cos_tile)


# Each run of this kernel is responsible for one input and a stride of sequence length in qk.
@triton.autotune(
    configs=[
        # Multiple configurations for triton to choose from. Triton will benchmark these and pick the best one.
        # Stages (typically 2-4): more stages allow for more overlapping of memory traffic with compute, but at the cost
        # of more SHMEM/register usage.
        # Warps (typically 2, 4, 8): more warps can increase occupancy, but at the cost of more SHMEM/register usage.
        triton.Config({"S_STRIDE": 64}, num_stages=2, num_warps=8),
        triton.Config({"S_STRIDE": 64}, num_stages=3, num_warps=8),
        triton.Config({"S_STRIDE": 256}, num_stages=2, num_warps=8),
        triton.Config({"S_STRIDE": 256}, num_stages=3, num_warps=8),
    ],
    key=["S"],
    restore_value=["qk_ptr"],  # Need to restore this when evaluating autotune configs because we update in-place.
)
@triton.jit
def _triton_yarn_kernel(
    qk_ptr: tl.pointer_type,  # [B, S, H, D] input/output tensor
    sin_ptr: tl.pointer_type,  # [S, D/2] sin tensor
    cos_ptr: tl.pointer_type,  # [S, D/2] cos tensor
    S: int,  # sequence length
    H: tl.constexpr,  # number of attention heads
    D_half: tl.constexpr,  # hidden dimension per head (half)
    s: float,  # scale factor
    stride_qkb: int,  # stride of qk tensor along the batch size
    stride_qks: int,  # stride of qk tensor along the sequence length
    stride_qkh: int,  # stride of qk tensor along the attention heads
    stride_qkd: int,  # stride of qk tensor along the hidden dimension
    stride_sins: int,  # stride of sin tensor along the sequence length
    stride_sind: int,  # stride of sin tensor along the hidden dimension
    stride_coss: int,  # stride of cos tensor along the sequence length
    stride_cosd: int,  # stride of cos tensor along the hidden dimension
    S_STRIDE: tl.constexpr,  # number of kernel instances along the sequence length
) -> None:
    pid_b = tl.program_id(axis=1)  # program ID in the second dimension
    pid_s = tl.program_id(axis=0)  # program ID in the first dimension

    # Note that we omit masking here because we assume H and D are always powers of 2.
    offsets_h = tl.arange(0, H)  # offsets along the attention heads
    offsets_d = tl.arange(0, 2 * D_half)  # offsets along hidden dimension
    offsets_d_half = tl.arange(0, D_half)  # offsets along half the hidden dimension

    # Iterate over rows assigned to this program ID. We stride the rows instead of processing contiguous groups of rows
    # because that leads to better memory access patterns across the kernel instances.
    for m in range(pid_s, S, S_STRIDE):
        # Load the sin and cos tiles for this sequence position m.
        sin_offsets = m * stride_sins + offsets_d_half * stride_sind  # [D/2] offsets
        cos_offsets = m * stride_coss + offsets_d_half * stride_cosd  # [D/2] offsets
        sin_tile = tl.load(sin_ptr + sin_offsets)  # [D/2] tile
        cos_tile = tl.load(cos_ptr + cos_offsets)  # [D/2] tile

        # Load the tile for this batch and sequence position m. Then split it into even and odd tiles.
        qk_offsets = (
            pid_b * stride_qkb
            + m * stride_qks
            + offsets_h[:, None] * stride_qkh
            + offsets_d[None, :] * stride_qkd
        )  # [H, D] offsets
        qk_tile = tl.load(qk_ptr + qk_offsets)  # [H, D] tile
        qk_0_tile, qk_1_tile = qk_tile.reshape(H, D_half, 2).split()  # [H, D/2] even and odd tiles

        # Apply rotations using element-wise multiplications between the qk tiles and sin/cos.
        qk_rot_0_tile = qk_0_tile * cos_tile[None, :] - qk_1_tile * sin_tile[None, :]  # [H, D/2] even tile
        qk_rot_1_tile = qk_0_tile * sin_tile[None, :] + qk_1_tile * cos_tile[None, :]  # [H, D/2] odd tile

        # Join the rotated tiles together, apply length scaling, and store the result back into the qk tensor.
        qk_rot_tile = tl.join(qk_rot_0_tile, qk_rot_1_tile).reshape(H, 2 * D_half)
        length_scale = 0.1 * tl.log(s) + 1
        tl.store(qk_ptr + qk_offsets, qk_rot_tile * length_scale)


def triton_yarn(
    qk: torch.Tensor,  # [B, S, 2, H, D] tensor containing queries and keys
    b: float,  # rotation base
    L: int,  # original context length
    s: float,  # scaling factor
    alpha: int,  # "far" threshold
    beta: int,   # "close" threshold
) -> torch.Tensor:
    B, S, _, H, D = qk.shape
    assert (H & (H - 1)) == 0, "H must be a power of 2"
    assert (D & (D - 1)) == 0, "D must be a power of 2"
    assert beta >= alpha, "Close threshold must be at least as high as far threshold"
    D_half = D // 2

    # First, launch a 1-D kernel that computes the sin/cos tensors. There is probably no benefit to writing this as a
    # triton kernel vs using torch directly, but this is for practice.
    sin = torch.empty(size=(S, D_half), device=DEVICE, dtype=qk.dtype)
    cos = torch.empty(size=(S, D_half), device=DEVICE, dtype=qk.dtype)

    # Compute alpha' and beta' outside of the triton kernel since it cannot access global functions.
    alpha_p, beta_p = _invert_r(alpha=alpha, beta=beta, D=D, b=b, L=L)

    sin_cos_grid = lambda META: (META["S_STRIDE"],)
    _triton_rope_sin_cos_kernel[sin_cos_grid](
        sin_ptr=sin,
        cos_ptr=cos,
        S=S,
        D_half=D_half,
        b=b,
        L=L,
        s=s,
        alpha_p=alpha_p,
        beta_p=beta_p,
        stride_sins=sin.stride(0),
        stride_sind=sin.stride(1),
        stride_coss=cos.stride(0),
        stride_cosd=cos.stride(1),
    )

    # Next, launch a 2-D kernel that actually applies the rotations.
    # We create a view on top of qk to collapse the query/key dimension with the head dimension, which allows us to
    # write the kernel in a more general way. It also slightly simplifies the offset computations we need to do inside
    # of the kernel which has a small performance benefit.
    qk_view = qk.view(B, S, 2 * H, D)

    rope_grid = lambda META: (META["S_STRIDE"], B)
    _triton_yarn_kernel[rope_grid](
        qk_ptr=qk_view,
        sin_ptr=sin,
        cos_ptr=cos,
        S=S,
        H=2*H,
        D_half=D_half,
        s=s,
        stride_qkb=qk_view.stride(0),
        stride_qks=qk_view.stride(1),
        stride_qkh=qk_view.stride(2),
        stride_qkd=qk_view.stride(3),
        stride_sins=sin.stride(0),
        stride_sind=sin.stride(1),
        stride_coss=cos.stride(0),
        stride_cosd=cos.stride(1),
    )

    return qk


def test() -> None:
    B = 16
    H = 16
    D = 256
    b = 10000
    s = 4
    alpha = 1
    beta = 32
    for S in [16, 64, 256]:
        L = S // s
        qk = torch.rand(size=(B, S, 2, H, D), device=DEVICE, dtype=torch.float32)
        rotated_vllm = vllm_yarn(qk.clone(), b=b, L=L, s=s, alpha=alpha, beta=beta)
        rotated_torch = torch_yarn(qk.clone(), b=b, L=L, s=s, alpha=alpha, beta=beta)
        rotated_torch_compile = torch_compile_yarn(qk.clone(), b=b, L=L, s=s, alpha=alpha, beta=beta)
        rotated_triton = triton_yarn(qk.clone(), b=b, L=L, s=s, alpha=alpha, beta=beta)

        # We use a higher tolerance of 1e-4 since the computation is fairly complex and more prone to small
        # numerical differences.
        assert torch.allclose(rotated_vllm, rotated_torch, atol=1e-4), \
            f"Test failure: mismatch for B={B}, S={S}, H={H}, D={D}, b={b}"
        assert torch.allclose(rotated_vllm, rotated_torch_compile, atol=1e-4), \
            f"Test failure: mismatch for B={B}, S={S}, H={H}, D={D}, b={b}"
        assert torch.allclose(rotated_vllm, rotated_triton, atol=1e-4), \
            f"Test failure: mismatch for B={B}, S={S}, H={H}, D={D}, b={b}"

    print("=========================")
    print("=== All tests passed! ===")
    print("=========================")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(6, 11)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["vllm", "torch", "torch_compile", "triton"],  # Possible values for `line_arg`.
        line_names=["vLLM", "Torch", "Torch (Compile)", "Triton"],  # Label name for the lines.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="yarn-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ),
)
def benchmark(S: int, provider: Literal["vllm", "torch", "torch_compile", "triton"]) -> float:
    B = 16
    H = 16
    D = 256
    b = 10000
    L = S // 4
    s = S / L
    alpha = 1
    beta = 32
    qk = torch.rand(size=(B, S, 2, H, D), device=DEVICE, dtype=torch.float32)
    if provider == "vllm":
        ms = cast("float", triton.testing.do_bench(fn=lambda: vllm_yarn(qk, b=b, L=L, s=s, alpha=alpha, beta=beta)))
    elif provider == "torch":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_yarn(qk, b=b, L=L, s=s, alpha=alpha, beta=beta)))
    elif provider == "torch_compile":
        ms = cast(
            "float",
            triton.testing.do_bench(fn=lambda: torch_compile_yarn(qk, b=b, L=L, s=s, alpha=alpha, beta=beta))
        )
    elif provider == "triton":
        ms = cast("float", triton.testing.do_bench(fn=lambda: triton_yarn(qk, b=b, L=L, s=s, alpha=alpha, beta=beta)))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Compute bytes processed: every element of qk is read and written once.
    bytes_processed = 2 * qk.numel() * qk.element_size()

    # Return bandwidth in GB/s.
    return bytes_processed * 1e-9 / (ms * 1e-3)


def main() -> None:
    # Enable Triton autotuning logging.
    import os
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    # Set default device to CUDA so all tensors are automatically created there
    torch.set_default_device(torch.device("cuda"))

    # Set reduced precision for float32 matmuls so we can leverage tensor cores.
    torch.set_float32_matmul_precision("high")

    print("Running YaRN benchmark on device:", DEVICE)

    # Run a simple correctness test first.
    test()

    # Run the benchmark.
    benchmark.run(print_data=True)

    # Results on an RTX 5080.
    #
    # yarn-performance:
    #         S        vLLM       Torch  Torch (Compile)      Triton
    # 0    64.0  307.695798  125.727618       483.748836  359.179331
    # 1   128.0  374.860524   88.472728       358.168273  396.326726
    # 2   256.0  538.861953   97.968358       405.979829  405.053464
    # 3   512.0  674.876659   99.372010       408.636860  435.806163
    # 4  1024.0  739.068983  100.029236       406.845851  448.267167
    #
    # Basically the same performance characteristics as regular RoPE.


if __name__ == "__main__":
    main()
