"""
Linear Layer Benchmarks

This script benchmarks the Triton implementation of nn.Linear against PyTorch's
implementation for both forward and backward passes.
"""

import torch
import time
import triton.testing
from linear import (
    linear_forward,
    linear_backward_input,
    linear_backward_weight,
    TritonLinear
)

def benchmark_forward(batch_size, in_features, out_features, num_runs=10):
    """Benchmark forward pass of PyTorch nn.Linear vs Triton implementation"""
    # Create input tensor
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)

    # Create weights
    weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float16)

    # Create PyTorch linear layer
    torch_linear = torch.nn.Linear(in_features, out_features, bias=False).to('cuda')
    torch_linear.weight.data = weight.clone()

    # Create Triton linear layer
    triton_linear = TritonLinear(in_features, out_features).to('cuda')
    triton_linear.weight.data = weight.clone()

    # Warmup
    torch_linear(x)
    triton_linear(x)

    # PyTorch benchmark
    torch.cuda.synchronize()
    pytorch_timer = time.perf_counter()
    for _ in range(num_runs):
        y_torch = torch_linear(x)
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - pytorch_timer) / num_runs

    # Triton benchmark
    torch.cuda.synchronize()
    triton_timer = time.perf_counter()
    for _ in range(num_runs):
        y_triton = triton_linear(x)
        torch.cuda.synchronize()
    triton_time = (time.perf_counter() - triton_timer) / num_runs

    # Convert to milliseconds
    pytorch_ms = pytorch_time * 1000
    triton_ms = triton_time * 1000
    speedup = pytorch_ms / triton_ms

    return pytorch_ms, triton_ms, speedup

def benchmark_backward_input(batch_size, in_features, out_features, num_runs=10):
    """Benchmark backward pass (input gradient) of PyTorch vs Triton"""
    # Create tensors
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16, requires_grad=True)
    weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float16)
    grad_output = torch.randn(batch_size, out_features, device='cuda', dtype=torch.float16)

    # PyTorch forward
    y = x @ weight.t()

    # Warmup
    y.backward(grad_output, retain_graph=True)
    x.grad.zero_()
    linear_backward_input(grad_output, weight)

    # PyTorch benchmark (only backward)
    torch.cuda.synchronize()
    pytorch_timer = time.perf_counter()
    for _ in range(num_runs):
        x.grad.zero_()
        y.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - pytorch_timer) / num_runs

    # Triton benchmark
    torch.cuda.synchronize()
    triton_timer = time.perf_counter()
    for _ in range(num_runs):
        grad_input = linear_backward_input(grad_output, weight)
        torch.cuda.synchronize()
    triton_time = (time.perf_counter() - triton_timer) / num_runs

    # Convert to milliseconds
    pytorch_ms = pytorch_time * 1000
    triton_ms = triton_time * 1000
    speedup = pytorch_ms / triton_ms

    return pytorch_ms, triton_ms, speedup

def benchmark_backward_weight(batch_size, in_features, out_features, num_runs=10):
    """Benchmark backward pass (weight gradient) of PyTorch vs Triton"""
    # Create tensors
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float16, requires_grad=True)
    grad_output = torch.randn(batch_size, out_features, device='cuda', dtype=torch.float16)

    # PyTorch forward
    y = x @ weight.t()

    # Warmup
    y.backward(grad_output, retain_graph=True)
    weight.grad.zero_()
    linear_backward_weight(x, grad_output)

    # PyTorch benchmark (only backward)
    torch.cuda.synchronize()
    pytorch_timer = time.perf_counter()
    for _ in range(num_runs):
        weight.grad.zero_()
        y.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - pytorch_timer) / num_runs

    # Triton benchmark
    torch.cuda.synchronize()
    triton_timer = time.perf_counter()
    for _ in range(num_runs):
        grad_weight = linear_backward_weight(x, grad_output)
        torch.cuda.synchronize()
    triton_time = (time.perf_counter() - triton_timer) / num_runs

    # Convert to milliseconds
    pytorch_ms = pytorch_time * 1000
    triton_ms = triton_time * 1000
    speedup = pytorch_ms / triton_ms

    return pytorch_ms, triton_ms, speedup

def run_all_benchmarks():
    """Run benchmarks for different sizes"""
    # Test different sizes
    sizes = [
        (128, 512, 512),
        (512, 1024, 1024),
        (1024, 2048, 1024),
        (2048, 4096, 2048),
    ]

    # Forward benchmarks
    print("\n=== Linear Forward Benchmarks ===")
    print(f"{'Batch Size':>10} | {'In Features':>11} | {'Out Features':>12} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>7}")
    print("-" * 80)

    for batch_size, in_features, out_features in sizes:
        pytorch_ms, triton_ms, speedup = benchmark_forward(batch_size, in_features, out_features)
        print(f"{batch_size:>10} | {in_features:>11} | {out_features:>12} | {pytorch_ms:>12.4f} | {triton_ms:>12.4f} | {speedup:>7.2f}x")

    # Backward input gradient benchmarks
    print("\n=== Linear Backward Input Gradient Benchmarks ===")
    print(f"{'Batch Size':>10} | {'In Features':>11} | {'Out Features':>12} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>7}")
    print("-" * 80)

    for batch_size, in_features, out_features in sizes:
        pytorch_ms, triton_ms, speedup = benchmark_backward_input(batch_size, in_features, out_features)
        print(f"{batch_size:>10} | {in_features:>11} | {out_features:>12} | {pytorch_ms:>12.4f} | {triton_ms:>12.4f} | {speedup:>7.2f}x")

    # Backward weight gradient benchmarks
    print("\n=== Linear Backward Weight Gradient Benchmarks ===")
    print(f"{'Batch Size':>10} | {'In Features':>11} | {'Out Features':>12} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>7}")
    print("-" * 80)

    for batch_size, in_features, out_features in sizes:
        pytorch_ms, triton_ms, speedup = benchmark_backward_weight(batch_size, in_features, out_features)
        print(f"{batch_size:>10} | {in_features:>11} | {out_features:>12} | {pytorch_ms:>12.4f} | {triton_ms:>12.4f} | {speedup:>7.2f}x")

if __name__ == "__main__":
    # Run all benchmarks
    print("Running benchmarks. This may take some time...")
    run_all_benchmarks()
