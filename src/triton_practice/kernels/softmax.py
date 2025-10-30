"""
Kernels for computing softmax along the last dimension (e.g. for LLM decoding).
"""

from typing import Literal, cast

import torch
import triton
import triton.language as tl
import triton.testing

from triton_practice.utils.device import DEVICE


def torch_native_softmax(X: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.softmax(X * scale, dim=-1)


def torch_softmax(X: torch.Tensor, scale: float) -> torch.Tensor:
    # Scale the input tensor and subtract the max for numerical stability.
    # Softmax is invariant to constant shifts.
    X_scaled = X * scale
    X_max = torch.max(X_scaled, dim=-1, keepdim=True).values
    X_scaled -= X_max

    # Compute softmax: exp(X) / sum(exp(X)).
    exp_X = torch.exp(X_scaled)
    sum_exp_X = torch.sum(exp_X, dim=-1, keepdim=True)
    return exp_X / sum_exp_X


torch_compile_softmax = torch.compile(torch_softmax)


# Each run of this kernel is responsible for a stride of rows in the input and output.
@triton.autotune(
    configs=[
        # Multiple configurations for triton to choose from. Triton will benchmark these and pick the best one.
        # Stages (typically 2-4): more stages allow for more overlapping of memory traffic with compute, but at the cost
        # of more SHMEM/register usage.
        # Warps (typically 2, 4, 8): more warps can increase occupancy, but at the cost of more SHMEM/register usage.
        triton.Config({"ROW_STRIDE": 64}, num_stages=2, num_warps=4),
        triton.Config({"ROW_STRIDE": 64}, num_stages=2, num_warps=8),
        triton.Config({"ROW_STRIDE": 256}, num_stages=2, num_warps=4),
        triton.Config({"ROW_STRIDE": 256}, num_stages=2, num_warps=8),
        triton.Config({"ROW_STRIDE": 1024}, num_stages=2, num_warps=4),
        triton.Config({"ROW_STRIDE": 1024}, num_stages=2, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _triton_softmax_kernel(
    X_ptr: tl.pointer_type,  # 1-D pointer to input X
    Y_ptr: tl.pointer_type,  # 1-D pointer to output Y
    scale: float,  # scaling factor
    M: int,  # number of rows in X and Y
    N: int,  # number of columns in X and Y
    N_PAD: tl.constexpr,  # number of columns in X and Y (padded to power of 2)
    stride_xm: int,  # stride of X along the rows
    stride_xn: int,  # stride of X along the columns
    stride_ym: int,  # stride of Y along the rows
    stride_yn: int,  # stride of Y along the columns
    ROW_STRIDE: tl.constexpr,  # number of kernel instances along the rows
) -> None:
    # Identify which block in the grid this kernel instance is responsible for.
    pid_m = tl.program_id(axis=0)  # program ID in the first dimension

    # Iterate over rows assigned to this program ID. We stride the rows instead of processing contiguous groups of rows
    # because that leads to better memory access patterns across the kernel instances.
    for m in range(pid_m, M, ROW_STRIDE):
        offsets_n = tl.arange(0, N_PAD)  # offsets along the columns
        mask = offsets_n < N  # mask to handle padding

        # Load the input row from X, apply scaling.
        X_offsets = m * stride_xm + offsets_n * stride_xn  # [N_PAD] offsets
        X_tile = tl.load(X_ptr + X_offsets, mask=mask, other=-float("inf"))  # masked [N_PAD] tile
        X_tile *= scale

        # Subtract the max for numerical stability. Softmax is invariant to constant shifts.
        X_tile -= tl.max(X_tile, axis=0)

        # Compute softmax: exp(X) / sum(exp(X)).
        exp_X = tl.exp(X_tile)
        sum_exp_X = tl.sum(exp_X, axis=0)
        Y_tile = exp_X / sum_exp_X

        # Store the output row to Y, applying the mask to avoid out-of-bounds writes.
        Y_offsets = m * stride_ym + offsets_n * stride_yn  # [N_PAD] offsets
        tl.store(Y_ptr + Y_offsets, Y_tile, mask=mask)


def triton_softmax(X: torch.Tensor, scale: float) -> torch.Tensor:
    assert X.dim() == 2, "Input tensor X must be 2-D"
    assert scale > 0.0, "Scale must be positive"

    M, N = X.shape
    N_PAD = triton.next_power_of_2(N)
    Y = torch.empty_like(X)

    # Create 1-D grid that launches one kernel per stride of rows.
    grid = lambda META: (META["ROW_STRIDE"],)

    # Launch the kernel with the grid.
    _triton_softmax_kernel[grid](
        X_ptr=X,
        Y_ptr=Y,
        scale=scale,
        M=M,
        N=N,
        N_PAD=N_PAD,
        stride_xm=X.stride(0),
        stride_xn=X.stride(1),
        stride_ym=Y.stride(0),
        stride_yn=Y.stride(1),
    )
    return Y


def test() -> None:
    sizes = [64, 256, 1024, 4096, 16384]
    for base_size in sizes:
        for delta in [-1, 0, 1]:
            size = base_size + delta
            scale = 1.0 + (size % 10) * 0.1  # vary scale for each size
            X = torch.rand(size=(size, size), device=DEVICE, dtype=torch.float32)
            Y_torch_native = torch_native_softmax(X, scale)
            Y_torch = torch_softmax(X, scale)
            Y_torch_compile = torch_compile_softmax(X, scale)
            Y_triton = triton_softmax(X, scale)
            assert torch.allclose(Y_torch_native, Y_torch, atol=1e-6), f"Test failure: mismatch for size {size}"
            assert torch.allclose(Y_torch_native, Y_torch_compile, atol=1e-6), f"Test failure: mismatch for size {size}"
            assert torch.allclose(Y_torch_native, Y_triton, atol=1e-6), f"Test failure: mismatch for size {size}"
    print("=========================")
    print("=== All tests passed! ===")
    print("=========================")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(9, 15)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["torch_native", "torch", "torch_compile", "triton"],  # Possible values for `line_arg`.
        line_names=["Torch (Native)", "Torch", "Torch (Compile)", "Triton"],  # Label name for the lines.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="softmax-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ),
)
def benchmark(size: int, provider: Literal["torch_native", "torch", "torch_compile", "triton"]) -> float:
    X = torch.rand(size=(size, size), device=DEVICE, dtype=torch.float32)
    scale = 1.0 + (size % 10) * 0.1  # vary scale for each size
    if provider == "torch_native":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_native_softmax(X, scale)))
    elif provider == "torch":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_softmax(X, scale)))
    elif provider == "torch_compile":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_compile_softmax(X, scale)))
    elif provider == "triton":
        ms = cast("float", triton.testing.do_bench(fn=lambda: triton_softmax(X, scale)))
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return 2 * X.numel() * X.element_size() * 1e-9 / (ms * 1e-3)


def main() -> None:
    # Enable Triton autotuning logging.
    import os
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    print("Running softmax benchmark on device:", DEVICE)

    # Run a simple correctness test first.
    test()

    # Run the benchmark.
    benchmark.run(print_data=True)

    # Results on an RTX 5080 (these numbers are high variance).
    #
    # softmax-performance:
    #       size  Torch (Native)       Torch  Torch (Compile)       Triton
    # 0    512.0      188.998544   70.969390       345.917422   283.405642
    # 1   1024.0      479.660608  208.981038       808.432927  1108.668629
    # 2   2048.0      459.560443  329.937929       924.696259   753.572387
    # 3   4096.0      407.667027  224.228759       544.785955   856.667700
    # 4   8192.0      413.665643  133.857561       447.314447   804.034247
    # 5  16384.0      410.632180  169.014781       379.541981   439.104900
    #
    # The bandwidth numbers for the triton kernel are good compared to the torch implementations. At 1024x1024, the
    # size of rows seems to interact well with the row-strided implementation, leading to efficient memory accesses.
    # On the other hand, once the rows get too big (16K * 4B = 64KB), the performance drops since we are operating on
    # a large amount of intermediate data, putting pressure on registers and limiting occupancy.
    #
    # The torch compiled version is actually better than the native version because the multiplication by scale is
    # fused into the softmax computation, reducing memory traffic (we can get the same effect by applying
    # torch.compile to the native version, but it's more interesting to look at this way).


if __name__ == "__main__":
    main()
