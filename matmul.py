import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A @ B
    A has shape (M, K), B has shape (K, N), C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Matrix multiplication kernel
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the current block
    # Load the pointers to the data for this block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Initialize accumulator to zero with float32 for higher precision during computation
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, using mask to handle boundary conditions
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K)
        # Convert inputs to float32 for the dot product
        # Compute the matrix multiplication of a and b
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert back to float16 before storing
    accumulator = accumulator.to(tl.float16)
    # Write back the block result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# Function to initialize and run the matrix multiplication
def matmul(a: torch.Tensor, b: torch.Tensor):
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    # Allocate output with float16
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Get grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Run the kernel
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
    )

    return c

def benchmark_matmul(M, N, K, num_runs=100):
    # Create input tensors with float16
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # Warmup
    torch.matmul(a, b)
    matmul(a, b)

    # PyTorch timing using do_bench
    ms_pytorch = triton.testing.do_bench(lambda: torch.matmul(a, b))

    # Triton timing using do_bench
    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))

    return ms_pytorch, ms_triton

def test_matmul():
    torch.manual_seed(0)

    # Test for correctness with float16
    a = torch.randn(128, 256, device='cuda', dtype=torch.float16)
    b = torch.randn(256, 128, device='cuda', dtype=torch.float16)

    c_torch = torch.matmul(a, b)
    c_triton = matmul(a, b)

    print("\nTesting correctness:")

    # Check for NaN values
    print(f"NaN in PyTorch result: {torch.isnan(c_torch).any()}")
    print(f"NaN in Triton result: {torch.isnan(c_triton).any()}")

    # Check for Inf values
    print(f"Inf in PyTorch result: {torch.isinf(c_torch).any()}")
    print(f"Inf in Triton result: {torch.isinf(c_triton).any()}")

    # Print matrix statistics
    print(f"\nInput matrix A - min: {a.min():.4f}, max: {a.max():.4f}, mean: {a.mean():.4f}")
    print(f"Input matrix B - min: {b.min():.4f}, max: {b.max():.4f}, mean: {b.mean():.4f}")
    print(f"PyTorch output - min: {c_torch.min():.4f}, max: {c_torch.max():.4f}, mean: {c_torch.mean():.4f}")
    print(f"Triton output - min: {c_triton.min():.4f}, max: {c_triton.max():.4f}, mean: {c_triton.mean():.4f}")

    # Compute errors
    abs_err = torch.abs(c_torch - c_triton)
    rel_err = torch.abs((c_torch - c_triton) / c_torch)

    print(f"\nAbsolute error - min: {abs_err.min():.4f}, max: {abs_err.max():.4f}, mean: {abs_err.mean():.4f}")
    print(f"Relative error - min: {rel_err.min():.4f}, max: {rel_err.max():.4f}, mean: {rel_err.mean():.4f}")

    # Original error metrics
    print(f"Max absolute error: {torch.max(abs_err):.4f}")
    print(f"Max relative error: {torch.max(rel_err):.4f}")

    # Benchmark different sizes
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    print("\nBenchmarking different matrix sizes:")
    print("Matrix Size (M=N=K) | PyTorch (ms) | Triton (ms) | Speedup")
    print("-" * 60)

    for M, N, K in sizes:
        ms_pytorch, ms_triton = benchmark_matmul(M, N, K)
        speedup = ms_pytorch / ms_triton
        print(f"{M:^16d} | {ms_pytorch:10.2f} | {ms_triton:10.2f} | {speedup:8.2f}x")

if __name__ == "__main__":
    test_matmul()
