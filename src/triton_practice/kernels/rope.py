"""
Kernels for computing rotary positional embeddings (RoPE) inside an attention layer.

Attention parameters:
B = batch size
S = sequence length
H = number of attention heads
D = hidden dimension per head

RoPE parameters:
b = rotation base

The RoPE operation does the following to each per-head query/key vector of size D (assumed even):
  1. Interpret each pair of adjacent dimensions as a 2D vector.
  2. Rotate each 2D vector by an angle that depends on the query/key's position m in the context.
     The rotation angle is determined by m * theta_d for the pair of indices (2*d, 2*d+1), where:
        theta_d = b ** (-2 * d / D)
  3. In practice, this means that x_{2d} and x_{2d+1} are updated as follows:
        x_{2d}   = x_{2d} * cos(m * theta_d) - x_{2d+1} * sin(m * theta_d)
        x_{2d+1} = x_{2d} * sin(m * theta_d) + x_{2d+1} * cos(m * theta_d)

For simplicity, we assume that the sequence length for all queries and keys is the same (S). Also, we don't worry
about caching th sin/cos tensors, which you would always want to do in practice. Lastly, the rotations are done on
a single tensor containing all queries, keys, and values, and they are applied to the input tensor in-place.
"""

from typing import Literal, cast

import torch
import triton
import triton.language as tl
import triton.testing
from vllm.vllm_flash_attn.layers.rotary import RotaryEmbedding

from triton_practice.utils.device import DEVICE


def vllm_rope(
    qkv: torch.Tensor,  # [B, S, 3, H, D] tensor containing queries, keys, and values
    b: float,  # rotation base
) -> torch.Tensor:
    D = qkv.shape[-1]
    assert D % 2 == 0, "D must be even"

    rope_embedding = RotaryEmbedding(dim=D, base=b, interleaved=True, device=DEVICE)
    return cast("torch.Tensor", rope_embedding.forward(qkv=qkv))


def torch_rope(
    qkv: torch.Tensor,  # [B, S, 3, H, D] tensor containing queries, keys, and values
    b: float,  # rotation base
) -> torch.Tensor:
    _, S, _, _, D = qkv.shape
    assert D % 2 == 0, "D must be even"

    # Compute the rotation angles for each (m, 2*d) pair: m * theta_d.
    position_ids = torch.arange(S, device=DEVICE, dtype=torch.float32)[:, None]  # [S, 1] indices
    dimensions = torch.arange(D//2, device=DEVICE, dtype=torch.float32)[None, :]  # [1, D/2] indices
    thetas = b ** (-2 * dimensions / D)  # [1, D/2] angles
    rotation_angles = torch.matmul(position_ids, thetas)  # [S, D/2] angles

    # Compute the rotation sin/cos.
    sin = torch.sin(rotation_angles).to(dtype=qkv.dtype)  # [S, D/2]
    cos = torch.cos(rotation_angles).to(dtype=qkv.dtype)  # [S, D/2]

    # Expand the shapes of sin/cos to enable broadcasting.
    sin = sin[None, :, None, None, :]  # [1, S, 1, 1, D/2]
    cos = cos[None, :, None, None, :]  # [1, S, 1, 1, D/2]

    # Extract just the query and key vectors (values do not get rotated).
    qk = qkv[:, :, 0:2, :, :]  # [B, S, 2, H, D]

    # Split qk tensor by the last dimension (hidden vector). qk_0 contains the even indices, and qk_1 the odd ones.
    qk_0 = qk[..., 0::2]  # [B, S, 2, H, D/2]
    qk_1 = qk[..., 1::2]  # [B, S, 2, H, D/2]

    # Apply rotations using element-wise multiplications between the qk splits and sin/cos. The sin/cos tensors are
    # automatically broadcast along the incomplete dimensions, so you can think of this as applying the [S, D/2]
    # versions of sin/cos to each batch/qk/head slice of the tensor.
    qk_rot_0 = qk_0 * cos - qk_1 * sin  # [B, S, 2, H, D/2]
    qk_rot_1 = qk_0 * sin + qk_1 * cos  # [B, S, 2, H, D/2]

    # Store results back into the qk tensor.
    qk[..., 0::2] = qk_rot_0
    qk[..., 1::2] = qk_rot_1

    return qkv


torch_compile_rope = torch.compile(torch_rope)


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

        # Compute the rotation angles for this row m and all d offsets.
        thetas = tl.exp(-tl.log(b) * offsets_d / D_half)  # [D/2] angles
        rotation_angles = m * thetas  # [D/2] angles

        # Compute sin and cos tiles.
        sin_tile = tl.sin(rotation_angles)  # [D/2] tile
        cos_tile = tl.cos(rotation_angles)  # [D/2] tile

        # Store the tiles into the sin and cos tensors at the right offsets.
        sin_offsets = m * stride_sins + offsets_d * stride_sind  # [D/2] offsets
        cos_offsets = m * stride_coss + offsets_d * stride_cosd  # [D/2] offsets
        tl.store(sin_ptr + sin_offsets, sin_tile)
        tl.store(cos_ptr + cos_offsets, cos_tile)


# Each run of this kernel is responsible for one batch and a stride of sequence length in qk.
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
def _triton_rope_kernel(
    qk_ptr: tl.pointer_type,  # [B, S, H, D] input/output tensor
    sin_ptr: tl.pointer_type,  # [S, D/2] sin tensor
    cos_ptr: tl.pointer_type,  # [S, D/2] cos tensor
    S: int,  # sequence length
    H: tl.constexpr,  # number of attention heads
    D_half: tl.constexpr,  # hidden dimension per head (half)
    stride_qkb: int,  # stride of qkv tensor along the batch size
    stride_qks: int,  # stride of qkv tensor along the sequence length
    stride_qkh: int,  # stride of qkv tensor along the attention heads
    stride_qkd: int,  # stride of qkv tensor along the hidden dimension
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

        # Join the rotated tiles together and store the result back into the qk tensor.
        qk_rot_tile = tl.join(qk_rot_0_tile, qk_rot_1_tile).reshape(H, 2 * D_half)
        tl.store(qk_ptr + qk_offsets, qk_rot_tile)


def triton_rope(
    qkv: torch.Tensor,  # [B, S, 3, H, D] tensor containing queries, keys, and values
    b: float,  # rotation base
) -> torch.Tensor:
    B, S, _, H, D = qkv.shape
    assert (H & (H - 1)) == 0, "H must be a power of 2"
    assert (D & (D - 1)) == 0, "D must be a power of 2"
    D_half = D // 2

    # First, launch a 1-D kernel that computes the sin/cos tensors. There is probably no benefit to writing this as a
    # triton kernel vs using torch directly, but this is for practice.
    sin = torch.empty(size=(S, D_half), device=DEVICE, dtype=qkv.dtype)
    cos = torch.empty(size=(S, D_half), device=DEVICE, dtype=qkv.dtype)

    sin_cos_grid = lambda META: (META["S_STRIDE"],)
    _triton_rope_sin_cos_kernel[sin_cos_grid](
        sin_ptr=sin,
        cos_ptr=cos,
        S=S,
        D_half=D_half,
        b=b,
        stride_sins=sin.stride(0),
        stride_sind=sin.stride(1),
        stride_coss=cos.stride(0),
        stride_cosd=cos.stride(1),
    )

    # Next, launch a 2-D kernel that actually applies the rotations.
    # We first extract just the queries and keys from qkv. And then we reshape to collapse the query/key dimension with
    # the head dimension, which allows us to write the kernel in a more general way. It also slightly simplifies the
    # offset computations we need to do inside of the kernel which has a small performance benefit.
    qk = qkv[:, :, 0:2, :, :].reshape(B, S, 2*H, D)
    assert qk.data_ptr() == qkv.data_ptr(), "qk must be a view of qkv"

    rope_grid = lambda META: (META["S_STRIDE"], B)
    _triton_rope_kernel[rope_grid](
        qk_ptr=qk,
        sin_ptr=sin,
        cos_ptr=cos,
        S=S,
        H=2*H,
        D_half=D_half,
        stride_qkb=qk.stride(0),
        stride_qks=qk.stride(1),
        stride_qkh=qk.stride(2),
        stride_qkd=qk.stride(3),
        stride_sins=sin.stride(0),
        stride_sind=sin.stride(1),
        stride_coss=cos.stride(0),
        stride_cosd=cos.stride(1),
    )

    return qkv


def test() -> None:
    B = 16
    H = 16
    D = 256
    b = 10000
    for S in [16, 64, 256]:
        qkv = torch.rand(size=(B, S, 3, H, D), device=DEVICE, dtype=torch.float32)
        rotated_vllm = vllm_rope(qkv.clone(), b=b)
        rotated_torch = torch_rope(qkv.clone(), b=b)
        rotated_torch_compile = torch_compile_rope(qkv.clone(), b=b)
        rotated_triton = triton_rope(qkv.clone(), b=b)

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
        plot_name="rope-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ),
)
def benchmark(S: int, provider: Literal["vllm", "torch", "torch_compile", "triton"]) -> float:
    B = 16
    H = 16
    D = 256
    b = 10000
    qkv = torch.rand(size=(B, S, 3, H, D), device=DEVICE, dtype=torch.float32)
    if provider == "vllm":
        ms = cast("float", triton.testing.do_bench(fn=lambda: vllm_rope(qkv, b=b)))
    elif provider == "torch":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_rope(qkv, b=b)))
    elif provider == "torch_compile":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_compile_rope(qkv, b=b)))
    elif provider == "triton":
        ms = cast("float", triton.testing.do_bench(fn=lambda: triton_rope(qkv, b=b)))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Compute bytes processed: every query/key element of qkv read and written once; values are untouched.
    bytes_processed = 2 * qkv.numel() * (2 / 3) * qkv.element_size()

    # Return bandwidth in GB/s.
    return bytes_processed * 1e-9 / (ms * 1e-3)


def main() -> None:
    # Enable Triton autotuning logging.
    import os
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    # Set reduced precision for float32 matmuls so we can leverage tensor cores.
    torch.set_float32_matmul_precision("high")

    print("Running RoPE benchmark on device:", DEVICE)

    # Run a simple correctness test first.
    test()

    # Run the benchmark.
    benchmark.run(print_data=True)

    # Results on an RTX 5080 (these numbers are high variance).
    #
    # rope-performance:
    #         S        vLLM       Torch  Torch (Compile)      Triton
    # 0    64.0  395.759061  161.878242       200.745826  383.118596
    # 1   128.0  582.869182  112.857348       174.267504  363.433328
    # 2   256.0  624.891514  112.462333       180.832109  427.916248
    # 3   512.0  729.906735  106.556511       182.638393  453.151756
    # 4  1024.0  763.896745  112.987397       184.565231  458.228789
    #
    # The triton implmeentation is much faster than the torch implementation even with compilation, probably due to the
    # fact that we don't need to materialize the intermediate rotated results before storing back into qkv (although
    # it's not clear why torch compilation doesn't fuse this -- maybe complexity from overwriting risk?). It still
    # falls short of the vLLM implementation, however, which uses its own optimized triton kernel inside.


if __name__ == "__main__":
    main()
