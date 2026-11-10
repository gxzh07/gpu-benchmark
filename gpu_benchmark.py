"""CPU vs NVIDIA GPU benchmark - run on Google Colab with a GPU runtime.

Colab: Runtime > Change runtime type > T4 GPU, then run:
    !python gpu_benchmark.py

Three experiments:
  1. A CUDA kernel written by hand (Numba) vs NumPy on the CPU  -> GPU programming
  2. Matrix multiply: NumPy (CPU) vs PyTorch on GPU (cuBLAS)    -> numeric libraries
  3. Neural-network inference across batch sizes, CPU vs GPU     -> parallel programming:
     finds the batch size where the GPU starts to win
"""

import copy
import math
import time

import numpy as np
import torch
from numba import cuda

assert torch.cuda.is_available(), "No GPU found. In Colab: Runtime > Change runtime type > T4 GPU"
GPU = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")


def time_it(fn, repeats=20, gpu=False):
    """Average time of fn() in milliseconds, after a warm-up run."""
    fn()  # warm-up: first GPU call includes one-time setup costs
    if gpu:
        torch.cuda.synchronize()  # GPU work is asynchronous: wait for it to finish
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if gpu:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


# ---------------------------------------------------------------- 1. hand-written CUDA kernel
@cuda.jit
def saxpy_kernel(a, x, y, out):
    """out = a*x + y. Each GPU thread computes ONE element, all in parallel."""
    i = cuda.grid(1)  # this thread's global index
    if i < out.size:  # extra threads at the end do nothing
        out[i] = a * x[i] + y[i]


print("1) Hand-written CUDA kernel (SAXPY: out = a*x + y)")
n = 50_000_000
x = np.random.rand(n).astype(np.float32)
y = np.random.rand(n).astype(np.float32)
d_x, d_y = cuda.to_device(x), cuda.to_device(y)  # copy once to GPU memory
d_out = cuda.device_array_like(d_x)
threads_per_block = 256
blocks = math.ceil(n / threads_per_block)


def gpu_saxpy():
    saxpy_kernel[blocks, threads_per_block](2.0, d_x, d_y, d_out)
    cuda.synchronize()


cpu_ms = time_it(lambda: 2.0 * x + y, repeats=5)
gpu_ms = time_it(gpu_saxpy, repeats=5)
assert np.allclose(d_out.copy_to_host(), 2.0 * x + y)  # check the kernel is correct
print(f"   {n:,} elements, {blocks:,} blocks x {threads_per_block} threads")
print(f"   CPU (NumPy): {cpu_ms:.2f} ms | GPU kernel: {gpu_ms:.2f} ms | speedup: {cpu_ms / gpu_ms:.1f}x\n")

# ---------------------------------------------------------------- 2. matrix multiply (cuBLAS)
print("2) Matrix multiply: NumPy on CPU vs PyTorch on GPU (cuBLAS)")
print(f"   {'size':>6} | {'CPU ms':>9} | {'GPU ms':>8} | {'speedup':>7} | {'GPU TFLOPS':>10}")
for size in [256, 1024, 2048, 4096]:
    a_cpu = np.random.rand(size, size).astype(np.float32)
    b_cpu = np.random.rand(size, size).astype(np.float32)
    a_gpu, b_gpu = torch.from_numpy(a_cpu).to(GPU), torch.from_numpy(b_cpu).to(GPU)
    cpu_ms = time_it(lambda: a_cpu @ b_cpu, repeats=3)
    gpu_ms = time_it(lambda: a_gpu @ b_gpu, gpu=True)
    tflops = 2 * size**3 / (gpu_ms / 1000) / 1e12  # a matmul is ~2*n^3 operations
    print(f"   {size:>6} | {cpu_ms:>9.2f} | {gpu_ms:>8.3f} | {cpu_ms / gpu_ms:>6.1f}x | {tflops:>10.2f}")
print()

# ---------------------------------------------------------------- 3. inference vs batch size
print("3) Neural-network inference: CPU vs GPU across batch sizes")
model_cpu = torch.nn.Sequential(  # 28 inputs, like the exoskeleton's sensors
    torch.nn.Linear(28, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 4),
).eval()
model_gpu = copy.deepcopy(model_cpu).to(GPU)  # same weights, on the GPU
print(f"   {'batch':>6} | {'CPU ms':>8} | {'GPU ms':>8} | {'GPU ms incl. copy':>17} | faster")
crossover = None
with torch.no_grad():
    for batch in [1, 8, 64, 512, 4096]:
        data_cpu = torch.randn(batch, 28)
        data_gpu = data_cpu.to(GPU)
        cpu_ms = time_it(lambda: model_cpu(data_cpu))
        gpu_ms = time_it(lambda: model_gpu(data_gpu), gpu=True)
        copy_ms = time_it(lambda: model_gpu(data_cpu.to(GPU)).cpu(), gpu=True)  # realistic
        winner = "GPU" if copy_ms < cpu_ms else "CPU"
        if winner == "GPU" and crossover is None:
            crossover = batch
        print(f"   {batch:>6} | {cpu_ms:>8.3f} | {gpu_ms:>8.3f} | {copy_ms:>17.3f} | {winner}")
print(f"\n   GPU starts winning (including data copies) at batch size: {crossover}")
