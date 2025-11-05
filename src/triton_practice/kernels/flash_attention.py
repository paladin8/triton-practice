"""
Kernels for computing flash attention.
See paper for details: https://arxiv.org/pdf/2205.14135

Attention parameters:
B = batch size
S = sequence length
Hq = number of query attention heads
Hk = number of key/value attention heads
D = hidden dimension per head

For simplicity, we will assume each sequence in the batch contains the same number of tokens, but we will support
multi-query attention (MQA) and grouped-query attention (GQA) setups where the number of key/value heads is less than
the number of query heads. In that scenario, query head h_q uses key/value head floor(h_q / g) where g = Hq / Hk.

Attention is defined as:
  A = softmax((Q * K^T) / sqrt(D) + mask) * V
which is a O(B * S^2 * H * D) computation that naively materializes O(B * S^2 * H + B * S * H * D) of memory. This is
expensive because S can be large (e.g. you want to support large contexts), and therefore you would be limited by
memory bandwidth of managing the intermediate results.

Flash attention is an algorithm that computes attention by iterating over chunks of keys and values such that you don't
need to materialize the entire Q * K^T matrix. It does this by computing a "streaming softmax."
  1. Assume we're working on one sequence and one attention head; re-define Q, K, V under that assumption. Also define:
        P = (Q * K^T) / sqrt(D) + mask
     for convenience. We apply a lower triangular mask to P for auto-regressive modeling (i.e. tokens should not be
     influenced by future tokens).
  2. Then define three temporary tensors m, l, and acc with one value per row of Q:
        m_i   = max(P_i)
        l_i   = sum(exp(P_i - m_i))
        acc_i = sum(exp(P_i - m_i) * V)
     which we will incrementally compute as we iterate over the rows of K and V.
  3. To process row i of Q and row j of K and V, we perform the following update:
        P_i       = (Q_i * K_j^T) / sqrt(D) + mask
        m_i_new   = max(m_i, P_{ij})
        l_i_new   = l_i * exp(m_i - m_i_new) + exp(P_{ij} - m_i_new)
        acc_i_new = acc_i * exp(m_i - m_i_new) + exp(P_i - m_i_new) * V_j
     In practice, we will perform this update on a row block of Q and streaming row blocks of K and V (not one row).
  4. Finally, once we've processed all of Q, K, V, we compute:
        A_i = acc_i / l_i
     to do row-level normalization of the accumulated sums.

This improves the performance of computing attention for two reasons: (a) we no longer need to materialize the O(S^2)
intermediate softmax matrix, and (b) we make good use of shared memory by loading blocks of Q, K, V at a time.
"""

import math
from typing import Literal, cast

import torch
import triton
import triton.language as tl
import triton.testing
from vllm.vllm_flash_attn import flash_attn_interface

from triton_practice.utils.device import DEVICE


def vllm_flash_attention(
    q: torch.Tensor,  # [B, S, Hq, D] tensor
    k: torch.Tensor,  # [B, S, Hk, D] tensor
    v: torch.Tensor,  # [B, S, Hk, D] tensor
) -> torch.Tensor:
    B, S, Hq, D = q.shape
    _, _, Hk, D = k.shape
    assert Hq % Hk == 0, "Number of query heads must be divisible by number of key/value heads"

    # vLLM expects q/k/v to have all tokens in the batch collapsed into the first dimension. It uses cumulative lengths
    # to identify the boundaries between sequences (in our case, the sequences are uniform in length).
    q = q.view(B * S, Hq, D)
    k = k.view(B * S, Hk, D)
    v = v.view(B * S, Hk, D)
    cu_seqlens = torch.arange(B + 1, device=q.device, dtype=torch.int32) * S

    out = cast(
        "torch.Tensor",
        flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=S,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=S,
            cu_seqlens_k=cu_seqlens,
            causal=True,  # apply causal mask for auto-regressive modeling
        )
    )
    return out.view(B, S, Hq, D)


def torch_flash_attention(
    q: torch.Tensor,  # [B, S, Hq, D] tensor
    k: torch.Tensor,  # [B, S, Hk, D] tensor
    v: torch.Tensor,  # [B, S, Hk, D] tensor
) -> torch.Tensor:
    B, S, Hq, D = q.shape
    _, _, Hk, D = k.shape
    assert (S & (S - 1)) == 0, "Sequence length must be a power of 2"
    assert Hq % Hk == 0, "Number of query heads must be divisible by number of key/value heads"

    # Block configuration.
    BLOCK_Q = BLOCK_KV = min(256, S)

    def _process_block(
        qb: torch.Tensor,  # [B * Hq, BLOCK_Q, D] tensor
        kb: torch.Tensor,  # [B * Hq, BLOCK_KV, D] tensor
        vb: torch.Tensor,  # [B * Hq, BLOCK_KV, D] tensor
        mask: torch.Tensor | None,  # [BLOCK_Q, BLOCK_KV] causal mask
        ms: torch.Tensor,   # [B * Hq, BLOCK_Q] tensor accumulating row maxes
        ls: torch.Tensor,   # [B * Hq, BLOCK_Q] tensor accumulating exp sums
        acc: torch.Tensor,  # [B * Hq, BLOCK_Q, D] tensor accumulating exp sum values
    ) -> None:
        # Compute attention softmax matrix (in fp32 for accumulation).
        P = torch.matmul(qb, kb.transpose(1, 2)) / math.sqrt(D)  # [B * Hq, BLOCK_Q, BLOCK_KV] tensor
        P.to(torch.float32)
        assert P.shape == (B * Hq, BLOCK_Q, BLOCK_KV), f"P block has wrong shape: {P.shape}"

        # Apply causal mask when when necessary.
        if mask is not None:
            P.masked_fill_(~mask[None, :, :], float("-inf"))

        # Update ms, ls, and acc in-place per the equations above.
        ms_block = torch.max(P, dim=-1)[0]  # [B * Hq, BLOCK_Q] tensor
        ms_new = torch.maximum(ms, ms_block)  # [B * Hq, BLOCK_Q] tensor
        scale_factor = torch.exp(ms - ms_new)  # [B * Hq, BLOCK_Q] tensor
        exp_P_scaled = torch.exp(P - ms_new[:, :, None])  # [B * Hq, BLOCK_Q, BLOCK_KV] tensor
        ls.mul_(scale_factor).add_(torch.sum(exp_P_scaled, dim=-1))
        acc.mul_(scale_factor[:, :, None]).add_(torch.matmul(exp_P_scaled.to(torch.float16), vb))
        ms.copy_(ms_new)

    # Reshape to a more suitable memory layout, and expand K/V as necessary to handle grouped-query attention case.
    # Expand is a zero-copy operation for repeating along singleton dimensions.
    qp = q.permute(0, 2, 1, 3).reshape(B * Hq, S, D)  # [B * Hq, S, D] tensor
    kp = k.permute(0, 2, 1, 3).unsqueeze(2).expand(B, Hk, Hq // Hk, S, D).reshape(B * Hq, S, D)  # [B * Hq, S, D] tensor
    vp = v.permute(0, 2, 1, 3).unsqueeze(2).expand(B, Hk, Hq // Hk, S, D).reshape(B * Hq, S, D)  # [B * Hq, S, D] tensor

    # Initialize temporary accumulation tensors that will be reused on every iteration over Q.
    ms = torch.zeros(size=(B * Hq, BLOCK_Q,), device=q.device, dtype=torch.float32)
    ls = torch.zeros(size=(B * Hq, BLOCK_Q), device=q.device, dtype=torch.float32)
    acc = torch.zeros(size=(B * Hq, BLOCK_Q, D), device=q.device, dtype=torch.float32)

    # Precompute single list of sequence indices that will be used to build the causal mask.
    indices = torch.arange(0, S, device=q.device)

    # Process the sequence length one block at a time.
    out = torch.zeros_like(q)
    for i in range(0, S, BLOCK_Q):
        qb = qp[:, i:i + BLOCK_Q, :]  # [B * Hq, BLOCK_Q, D] tensor
        assert qb.shape == (B * Hq, BLOCK_Q, D), f"Q block has wrong shape: {qb.shape}"

        # Reset temporary accumulation tensors.
        ms.fill_(float("-inf"))
        ls.fill_(1.0)
        acc.fill_(0)

        # Compute Q indices for causal mask, shared during the processing of the whole Q block.
        q_indices = indices[i:i + BLOCK_Q]

        # Only iterate K/V blocks until i + BLOCKQ because beyond that it's all masked out.
        for j in range(0, i + BLOCK_Q, BLOCK_KV):
            kb = kp[:, j:j + BLOCK_KV, :]  # [B * Hq, BLOCK_KV, D] tensor
            assert kb.shape == (B * Hq, BLOCK_KV, D), f"K block has wrong shape: {kb.shape}"
            vb = vp[:, j:j + BLOCK_KV, :]  # [B * Hq, BLOCK_KV, D] tensor
            assert vb.shape == (B * Hq, BLOCK_KV, D), f"V block has wrong shape: {vb.shape}"

            # Build causal mask based on q_indices and k_indices, i.e. preserving only positions where the
            # sequence index in Q is greater than the sequence index in K.
            if j + BLOCK_KV <= i:
                # Fully before diagonal, no mask necessary.
                mask = None
            else:
                # On the diagonal, compute the mask.
                kv_indices = indices[j:j + BLOCK_KV]
                mask = q_indices[:, None] >= kv_indices[None, :]

            _process_block(
                qb=qb,
                kb=kb,
                vb=vb,
                mask=mask,
                ms=ms,
                ls=ls,
                acc=acc,
            )

        # Compute attention result, reshape/permute it back, and set in output.
        block_out = acc / ls[:, :, None]  # [B * Hq, BLOCK_Q, D] tensor
        out[:, i:i + BLOCK_Q, :, :] = block_out.reshape(B, Hq, BLOCK_Q, D).permute(0, 2, 1, 3)

    return out


torch_compile_flash_attention = torch.compile(torch_flash_attention)


# Each run of this kernel is responsible for one block of rows from Q for a single input and query head.
@triton.autotune(
    configs=[
        # Multiple configurations for triton to choose from. Triton will benchmark these and pick the best one.
        # Stages (typically 2-4): more stages allow for more overlapping of memory traffic with compute, but at the cost
        # of more SHMEM/register usage.
        # Warps (typically 2, 4, 8): more warps can increase occupancy, but at the cost of more SHMEM/register usage.
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 128}, num_stages=1, num_warps=4),
    ],
    key=["S", "D"],
)
@triton.jit
def _triton_flash_attention_kernel(
    q_ptr: tl.pointer_type,  # [B, Hq, S, D] tensor
    k_ptr: tl.pointer_type,  # [B, Hk, S, D] tensor
    v_ptr: tl.pointer_type,  # [B, Hk, S, D] tensor
    out_ptr: tl.pointer_type,  # [B, Hq, S, D] tensor
    S: tl.constexpr,  # sequence length
    D: tl.constexpr,  # hidden dimension
    qk_scale: tl.float16,  # scale QK by this (log_2(e) / sqrt(D))  # type: ignore
    group_size: int,  # ratio of query heads to key/value heads
    stride_qb: int,  # stride of q_ptr along batch size
    stride_qh: int,  # stride of q_ptr along attention heads
    stride_qs: int,  # stride of q_ptr along sequence length
    stride_qd: int,  # stride of q_ptr along hidden dimension
    stride_kb: int,  # stride of k_ptr along batch size
    stride_kh: int,  # stride of k_ptr along attention heads
    stride_ks: int,  # stride of k_ptr along sequence length
    stride_kd: int,  # stride of k_ptr along hidden dimension
    stride_vb: int,  # stride of v_ptr along batch size
    stride_vh: int,  # stride of v_ptr along attention heads
    stride_vs: int,  # stride of v_ptr along sequence length
    stride_vd: int,  # stride of v_ptr along hidden dimension
    stride_outb: int,  # stride of out_ptr along batch size
    stride_outh: int,  # stride of out_ptr along attention heads
    stride_outs: int,  # stride of out_ptr along sequence length
    stride_outd: int,  # stride of out_ptr along hidden dimension
    BLOCK_Q: tl.constexpr,   # block size of rows of Q to process
    BLOCK_KV: tl.constexpr,  # block size of rows of K/V to process
) -> None:
    pid_b = tl.program_id(axis=2)  # Program ID for batch
    pid_h = tl.program_id(axis=1)  # Program ID for query head
    pid_q = tl.program_id(axis=0)  # Program ID for block of rows from Q

    # Compute key/value head from query head.
    h_kv = pid_h // group_size

    # Construct block pointers to allow triton to optimize loads better. Q and V blocks are loaded normally while K
    # blocks are loaded transposed for the matmul.
    q_start = pid_q * BLOCK_Q
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(S, D),
        strides=(stride_qs, stride_qd),
        offsets=(q_start, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + h_kv * stride_kh,
        shape=(D, S),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(D, BLOCK_KV),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + h_kv * stride_vh,
        shape=(S, D),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, D),
        order=(1, 0),
    )

    # Compute indices and load block from Q.
    q_indices = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q] tensor
    qb = tl.load(q_block_ptr, boundary_check=(0,))  # [BLOCK_Q, D] tensor

    # Initialize temporary accumulation tensors.
    ms = tl.full(shape=(BLOCK_Q,), value=float("-inf"), dtype=tl.float32)  # [BLOCK_Q] tensor
    ls = tl.full(shape=(BLOCK_Q,), value=1.0, dtype=tl.float32)  # [BLOCK_Q] tensor
    acc = tl.zeros(shape=(BLOCK_Q, D), dtype=tl.float32)  # [BLOCK_Q, D] tensor

    # Iterate over blocks of rows of K/V until q_start + BLOCKQ because beyond that it's all masked out.
    for kv_start in range(0, q_start + BLOCK_Q, BLOCK_KV):
        # Load block from K (defer V until later when we need it).
        kb = tl.load(k_block_ptr, boundary_check=(1,))  # [D, BLOCK_KV] tensor

        # Compute attention softmax matrix.
        P = tl.dot(qb, kb, out_dtype=tl.float32)  # [BLOCK_Q, BLOCK_KV] tensor
        P *= qk_scale

        # Apply causal mask when when necessary.
        if kv_start + BLOCK_KV >= q_start:
            kv_indices = kv_start + tl.arange(0, BLOCK_KV)  # [BLOCK_KV] offsets
            mask = q_indices[:, None] >= kv_indices[None, :]  # [BLOCK_Q, BLOCK_KV] mask
            P = tl.where(mask, P, float("-inf"))

        # Compute intermediate quantities that will be used in ms, ls, and acc updates. Note that exp2 is used here in
        # place of exp because we apply a scaling factor outside of the kernel (see qk_scale definition).
        ms_new = tl.maximum(ms, tl.max(P, axis=-1))  # [BLOCK_Q] tensor
        P -= ms_new[:, None]
        exp_P_scaled = tl.exp2(P)  # [BLOCK_Q, BLOCK_KV] tensor
        scale_factor = tl.exp2(ms - ms_new)  # [BLOCK_Q] tensor

        # Load block from V just before using it.
        vb = tl.load(v_block_ptr, boundary_check=(0,))  # [BLOCK_KV, D] tensor

        # Update ms, ls, and acc per the equations above.
        ls = scale_factor * ls + tl.sum(exp_P_scaled, axis=-1)
        acc = scale_factor[:, None] * acc + tl.dot(exp_P_scaled.to(tl.float16), vb, out_dtype=tl.float32)
        ms = ms_new

        # Advance block pointers.
        k_block_ptr = tl.advance(base=k_block_ptr, offsets=(0, BLOCK_KV))
        v_block_ptr = tl.advance(base=v_block_ptr, offsets=(BLOCK_KV, 0))

    # Compute attention result and set in output.
    acc /= ls[:, None]
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + pid_b * stride_outb + pid_h * stride_outh,
        shape=(S, D),
        strides=(stride_outs, stride_outd),
        offsets=(q_start, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    tl.store(out_block_ptr, acc.to(tl.float16), boundary_check=(0,))


def triton_flash_attention(
    q: torch.Tensor,  # [B, S, Hq, D] tensor
    k: torch.Tensor,  # [B, S, Hk, D] tensor
    v: torch.Tensor,  # [B, S, Hk, D] tensor
) -> torch.Tensor:
    B, S, Hq, D = q.shape
    _, _, Hk, D = k.shape
    assert (S & (S - 1)) == 0, "Sequence length must be a power of 2"
    assert (D & (D - 1)) == 0, "Hidden dimension must be a power of 2"
    assert Hq % Hk == 0, "Number of query heads must be divisible by number of key/value heads"

    # Create output tensor.
    out = torch.zeros_like(q)  # [B, S, Hq, D] tensor

    # Optimization: we prefer to use exp2 instead of exp inside the kernel, as GPUs have a hardware intrinsic for exp2.
    # So we multiply QK by log_2(e) here on top of the 1 / sqrt(D) factor from the attention formula, which correctly
    # adjusts for the modified exponentiation base.
    qk_scale = 1.44269504089 / math.sqrt(D)

    # Launch a 3-D kernel per (block of rows in Q, query head, batch).
    grid = lambda META: (triton.cdiv(S, META["BLOCK_Q"]), Hq, B)
    _triton_flash_attention_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        out_ptr=out,
        S=S,
        D=D,
        group_size=Hq // Hk,
        qk_scale=qk_scale,
        # These strides permute dimensions 1 and 2 in accordance with the dimensions that the kernel expects. This
        # bypasses the need to reshape these tensors before passing them in.
        stride_qb=q.stride(0),
        stride_qh=q.stride(2),
        stride_qs=q.stride(1),
        stride_qd=q.stride(3),
        stride_kb=k.stride(0),
        stride_kh=k.stride(2),
        stride_ks=k.stride(1),
        stride_kd=k.stride(3),
        stride_vb=v.stride(0),
        stride_vh=v.stride(2),
        stride_vs=v.stride(1),
        stride_vd=v.stride(3),
        stride_outb=out.stride(0),
        stride_outh=out.stride(2),
        stride_outs=out.stride(1),
        stride_outd=out.stride(3),
    )

    return out



def test() -> None:
    B = 4
    Hq = 64
    Hk = 8
    D = 64
    for S in [64, 256, 1024]:
        q = torch.rand(size=(B, S, Hq, D), device=DEVICE, dtype=torch.float16)
        k = torch.rand(size=(B, S, Hk, D), device=DEVICE, dtype=torch.float16)
        v = torch.rand(size=(B, S, Hk, D), device=DEVICE, dtype=torch.float16)
        out_vllm = vllm_flash_attention(q=q, k=k, v=v)
        out_torch = torch_flash_attention(q=q, k=k, v=v)
        out_torch_compile = torch_compile_flash_attention(q=q, k=k, v=v)
        out_triton = triton_flash_attention(q=q, k=k, v=v)

        diff_torch = (out_vllm - out_torch).abs().max().item()
        diff_torch_compile = (out_vllm - out_torch_compile).abs().max().item()
        diff_triton = (out_vllm - out_triton).abs().max().item()

        # We use a higher tolerance of 1e-3 since some of the computations are done using fp16.
        assert torch.allclose(out_vllm, out_torch, atol=1e-3), \
            f"Test failure: mismatch for B={B}, S={S}, Hq={Hq}, Hk={Hk}, D={D}, diff={diff_torch}"
        assert torch.allclose(out_vllm, out_torch_compile, atol=1e-3), \
            f"Test failure: mismatch for B={B}, S={S}, Hq={Hq}, Hk={Hk}, D={D}, diff={diff_torch_compile}"
        assert torch.allclose(out_vllm, out_triton, atol=1e-3), \
            f"Test failure: mismatch for B={B}, S={S}, Hq={Hq}, Hk={Hk}, D={D}, diff={diff_triton}"

    print("=========================")
    print("=== All tests passed! ===")
    print("=========================")


def profile() -> None:
    B = 4
    Hq = 64
    Hk = 8
    D = 64
    S = 4096
    q = torch.rand(size=(B, S, Hq, D), device=DEVICE, dtype=torch.float16)
    k = torch.rand(size=(B, S, Hk, D), device=DEVICE, dtype=torch.float16)
    v = torch.rand(size=(B, S, Hk, D), device=DEVICE, dtype=torch.float16)

    # Do 10 warmup iterations.
    for _ in range(10):
        triton_flash_attention(q=q, k=k, v=v)

    # Profile 10 real iterations.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        for _ in range(10):
            triton_flash_attention(q=q, k=k, v=v)

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(8, 13)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["vllm", "torch", "torch_compile", "triton"],  # Possible values for `line_arg`.
        line_names=["vLLM", "Torch", "Torch (Compile)", "Triton"],  # Label name for the lines.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="flash-attention-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ),
)
def benchmark(S: int, provider: Literal["vllm", "torch", "torch_compile", "triton"]) -> float:
    B = 4
    Hq = 64
    Hk = 8
    D = 64
    q = torch.rand(size=(B, S, Hq, D), device=DEVICE, dtype=torch.float16)
    k = torch.rand(size=(B, S, Hk, D), device=DEVICE, dtype=torch.float16)
    v = torch.rand(size=(B, S, Hk, D), device=DEVICE, dtype=torch.float16)
    if provider == "vllm":
        ms = cast("float", triton.testing.do_bench(fn=lambda: vllm_flash_attention(q=q, k=k, v=v)))
    elif provider == "torch":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_flash_attention(q=q, k=k, v=v)))
    elif provider == "torch_compile":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_compile_flash_attention(q=q, k=k, v=v)))
    elif provider == "triton":
        ms = cast("float", triton.testing.do_bench(fn=lambda: triton_flash_attention(q=q, k=k, v=v)))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Compute bytes processed: every element of q, k, and v is read once, an output the same size of q is written.
    bytes_processed = 2 * q.numel() * q.element_size() + k.numel() * k.element_size() + v.numel() * v.element_size()

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

    print("Running flash attention benchmark on device:", DEVICE)

    # Run a simple correctness test first.
    test()

    # Run repeatedly to profile.
    # profile()

    # Run the benchmark.
    benchmark.run(print_data=True)

    # Results on an RTX 5080.
    #
    # flash-attention-performance:
    #         S        vLLM      Torch  Torch (Compile)      Triton
    # 0   256.0  290.594383  18.899256        55.626485  114.792367
    # 1   512.0  269.460296  13.770371        44.777605   92.884442
    # 2  1024.0  186.331160   8.849901        31.512265   71.069269
    # 3  2048.0  112.127308   5.169715        19.390696   46.505103
    # 4  4096.0   61.018511   2.822685        10.357715   27.181299
    #
    # Both the torch and torch compiled implementations are quite slow here, as they have a harder time taking
    # advantage of the design of flash attention (+ I didn't spend time optimizing them). While the triton
    # implementation is much faster than torch, despite my best efforts to optimize it, it is still 2-3x slower than
    # the vLLM implementation.


if __name__ == "__main__":
    main()
