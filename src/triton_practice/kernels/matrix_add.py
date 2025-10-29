from typing import Literal, cast

import torch
import triton
import triton.language as tl
import triton.testing

from triton_practice.utils.device import DEVICE


def torch_matrix_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape == B.shape, "Input tensors must have the same shape"
    return A + B


# Each run of this kernel is responsible for a [BLOCK_M, BLOCK_N] tile of the output matrix C.
@triton.autotune(
    configs=[
        # Multiple configurations for triton to choose from. Triton will benchmark these and pick the best one.
        # Stages (typically 2-4): more stages allow for more overlapping of memory traffic with compute, but at the cost
        # of more SHMEM/register usage.
        # Warps (typically 2, 4, 8): more warps can increase occupancy, but at the cost of more SHMEM/register usage.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _matrix_add_kernel_2d(
    A_ptr: tl.pointer_type,  # 1-D pointer to beginning of matrix A
    B_ptr: tl.pointer_type,  # 1-D pointer to beginning of matrix B
    C_ptr: tl.pointer_type,  # 1-D pointer to beginning of output matrix C
    M: int,  # number of rows in A, B, and C
    N: int,  # number of columns in A, B, and C
    stride_am: int,  # stride between consecutive rows of A (i.e. # elements to skip in A_ptr to get to the next row)
    stride_an: int,  # stride between consecutive columns of A (should be 1 for row-major)
    stride_bm: int,  # stride between consecutive rows of B
    stride_bn: int,  # stride between consecutive columns of B
    stride_cm: int,  # stride between consecutive rows of C
    stride_cn: int,  # stride between consecutive columns of C
    BLOCK_M: tl.constexpr,  # number of rows in each block
    BLOCK_N: tl.constexpr,  # number of columns in each block
) -> None:
    # Identify which point in the grid this kernel instance is responsible for.
    pid_m = tl.program_id(axis=0)  # program ID in the first dimension of the grid
    pid_n = tl.program_id(axis=1)  # program ID in the second dimension of the grid

    # Construct the 1-D ranges of row and column offsets for this block.
    offsets_m = pid_m * BLOCK_M + tl.arange(start=0, end=BLOCK_M)  # [BLOCK_M] range of row offsets for this block
    offsets_n = pid_n * BLOCK_N + tl.arange(start=0, end=BLOCK_N)  # [BLOCK_N] range of column offsets for this block

    # Use broadcasting to create the 2-D offsets for all elements in the [BLOCK_M, BLOCK_N] tile.
    # The [BLOCK_M] range offs_m is repeated for BLOCK_N columns to turn into a [BLOCK_M, BLOCK_N] array of row offsets.
    # The [BLOCK_N] range offs_n is repeated for BLOCK_M rows to turn into a [BLOCK_M, BLOCK_N] array of column offsets.
    # We multiply the row and column offsets by their respective strides to get the final 2-D offsets into the 1-D
    # representations of the matrices.
    A_offsets = offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an  # 2-D offsets for A
    B_offsets = offsets_m[:, None] * stride_bm + offsets_n[None, :] * stride_bn  # 2-D offsets for B
    C_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn  # 2-D offsets for C

    # Compute the mask for handling elements within bounds. This matters when M/N are not multiples of BLOCK_M/BLOCK_N.
    # This mask is a [BLOCK_M, BLOCK_N] array where each element is True if the row and column offset is within [M, N].
    # Again we repeat offs_m and offs_n using broadcasting so the end result is [BLOCK_M, BLOCK_N].
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)  # [BLOCK_M, BLOCK_N] mask

    # Load the [BLOCK_M, BLOCK_N] tiles from A and B, applying the mask and using 0.0 for out-of-bounds elements.
    A_tile = tl.load(pointer=A_ptr + A_offsets, mask=mask, other=0.0)  # masked [BLOCK_M, BLOCK_N] tile from A
    B_tile = tl.load(pointer=B_ptr + B_offsets, mask=mask, other=0.0)  # masked [BLOCK_M, BLOCK_N] tile from B

    # Perform the computation. This just represents the addition and does not allocate intermediate memory.
    C_tile = A_tile + B_tile  # [BLOCK_M, BLOCK_N] tile of C

    # Store the result tile back to C, applying the mask to avoid out-of-bounds writes.
    tl.store(pointer=C_ptr + C_offsets, value=C_tile, mask=mask)


def triton_matrix_add_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape == B.shape, "Input tensors must have the same shape"
    assert A.dim() == 2, "Input tensors must be 2-D matrices"

    M, N = A.shape  # A is a [M, N] matrix
    C = torch.empty_like(A)  # Empty [M, N] output matrix

    # Create 2-D grid that launches one kernel instance per [BLOCK_M, BLOCK_N] tile of the output matrix C.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    # Launch the kernel with the grid.
    _matrix_add_kernel_2d[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        stride_am=A.stride(0),
        stride_an=A.stride(1),
        stride_bm=B.stride(0),
        stride_bn=B.stride(1),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
    )

    return C


# Each run of this kernel is responsible for a [BLOCK_SIZE] tile of the output matrix C.
@triton.autotune(
    configs=[
        # Multiple configurations for triton to choose from. Triton will benchmark these and pick the best one.
        # Stages (typically 2-4): more stages allow for more overlapping of memory traffic with compute, but at the cost
        # of more SHMEM/register usage.
        # Warps (typically 2, 4, 8): more warps can increase occupancy, but at the cost of more SHMEM/register usage.
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=2, num_warps=8),
    ],
    key=["N_ELEMS"],
)
@triton.jit
def _matrix_add_kernel_1d(
    A_ptr: tl.pointer_type,  # 1-D pointer to beginning of matrix A
    B_ptr: tl.pointer_type,  # 1-D pointer to beginning of matrix B
    C_ptr: tl.pointer_type,  # 1-D pointer to beginning of output matrix C
    N_ELEMS: int,  # number of elements in A, B, and C
    BLOCK_SIZE: tl.constexpr,  # number of elements in each block
) -> None:
    # Identify which point in the grid this kernel instance is responsible for.
    pid = tl.program_id(axis=0)  # program ID in the grid

    # Construct the 1-D range of offsets for this block.
    offsets = pid * BLOCK_SIZE + tl.arange(start=0, end=BLOCK_SIZE)  # [BLOCK_SIZE] range of offsets for this block

    # Compute the mask for handling elements within bounds. This matters when N_ELEMS is not a multiple of BLOCK_SIZE.
    mask = offsets < N_ELEMS  # [BLOCK_SIZE] mask

    # Load the [BLOCK_SIZE] blocks from A and B, applying the mask and using 0.0 for out-of-bounds elements.
    A_block = tl.load(pointer=A_ptr + offsets, mask=mask, other=0.0)  # masked [BLOCK_SIZE] block from A
    B_block = tl.load(pointer=B_ptr + offsets, mask=mask, other=0.0)  # masked [BLOCK_SIZE] block from B

    # Perform the computation. This just represents the addition and does not allocate intermediate memory.
    C_block = A_block + B_block  # [BLOCK_SIZE] block of C

    # Store the result block back to C, applying the mask to avoid out-of-bounds writes.
    tl.store(pointer=C_ptr + offsets, value=C_block, mask=mask)


def triton_matrix_add_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape == B.shape, "Input tensors must have the same shape"
    assert A.dim() == 2, "Input tensors must be 2-D matrices"

    # Note that this version will only be efficient if A and B are already stored contiguously in memory.
    A_flat = A.contiguous().view(-1)  # Flatten A to 1-D
    B_flat = B.contiguous().view(-1)  # Flatten B to 1-D
    N_ELEMS = A.numel()  # Total number of elements in A and B

    C_flat = torch.empty_like(A_flat)  # Empty [M, N] output matrix

    # Create 1-D grid that launches one kernel instance per [BLOCK_SIZE] elements of the output matrix C.
    grid = lambda META: (triton.cdiv(N_ELEMS, META["BLOCK_SIZE"]),)

    # Launch the kernel with the grid.
    _matrix_add_kernel_1d[grid](
        A_ptr=A_flat,
        B_ptr=B_flat,
        C_ptr=C_flat,
        N_ELEMS=N_ELEMS,
    )

    return C_flat.view_as(A)  # Reshape C back to [M, N]


def test() -> None:
    sizes = [128, 256, 512, 1024]
    for size in sizes:
        A = torch.rand(size=(size, size), device=DEVICE, dtype=torch.float32)
        B = torch.rand(size=(size, size), device=DEVICE, dtype=torch.float32)
        C_torch = torch_matrix_add(A, B)
        C_triton_2d = triton_matrix_add_2d(A, B)
        C_triton_1d = triton_matrix_add_1d(A, B)
        assert torch.allclose(C_torch, C_triton_2d), f"Test failure: mismatch for size {size}"
        assert torch.allclose(C_torch, C_triton_1d), f"Test failure: mismatch for size {size}"
    print("=========================")
    print("=== All tests passed! ===")
    print("=========================")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(9, 15)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["torch", "triton2d", "triton1d"],  # Possible values for `line_arg`.
        line_names=["Torch", "Triton (2-D)", "Triton (1-D)"],  # Label name for the lines.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="matrix-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ),
)
def benchmark(size: int, provider: Literal["torch", "triton2d", "triton1d"]) -> float:
    A = torch.rand(size=(size, size), device=DEVICE, dtype=torch.float32)
    B = torch.rand(size=(size, size), device=DEVICE, dtype=torch.float32)
    if provider == "torch":
        ms = cast("float", triton.testing.do_bench(fn=lambda: torch_matrix_add(A, B)))
    elif provider == "triton2d":
        ms = cast("float", triton.testing.do_bench(fn=lambda: triton_matrix_add_2d(A, B)))
    elif provider == "triton1d":
        ms = cast("float", triton.testing.do_bench(fn=lambda: triton_matrix_add_1d(A, B)))
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return 3 * A.numel() * A.element_size() * 1e-9 / (ms * 1e-3)


def main() -> None:
    # Enable Triton autotuning logging.
    import os
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    print("Running matrix_add benchmark on device:", DEVICE)

    # Run a simple correctness test first.
    test()

    # Run the benchmark.
    benchmark.run(print_data=True)

    # Results on an RTX 5080.
    #
    # matrix-add-performance:
    #       size       Torch  Triton (2-D)  Triton (1-D)
    # 0    512.0  371.408985    289.331248    362.659802
    # 1   1024.0  714.219807    388.032560    710.301425
    # 2   2048.0  783.232646    485.896088    778.138171
    # 3   4096.0  817.348086    553.718493    803.120682
    # 4   8192.0  812.700535    615.967132    848.483353
    # 5  16384.0  839.159456    623.336654    852.714630
    #
    # The 1-D version performs better than the 2-D version because it assumes the input is contiguous in memory, which
    # is true for the matrices here. In the 1-D kernel, the memory access patterns are better coalesced under this
    # assumption (traverses the memory block fully sequentially instead of "jumping" to handle tiles across multiple
    # rows). The 2-D kernel deals with arbitrary strides and puts pressure on registers while computing all of the
    # offsets, which makes it less efficient but more general.


if __name__ == "__main__":
    main()
