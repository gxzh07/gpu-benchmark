# GPU Performance Benchmark: CPU vs NVIDIA GPU

Three experiments measuring when, and why, an NVIDIA GPU beats a CPU:

1. **A hand-written CUDA kernel** (in Python with Numba) vs NumPy on the CPU
2. **Matrix multiplication** with NumPy on the CPU vs PyTorch on the GPU (cuBLAS)
3. **Neural-network inference** across batch sizes, to find where the GPU starts to win

Run on Google Colab with an **NVIDIA Tesla T4** GPU.

## Results

### 1. Hand-written CUDA kernel: SAXPY (`out = a*x + y`)

Each GPU thread computes one element. 50,000,000 elements run as 195,313 blocks x 256 threads.

| | Time | Speedup |
|---|---|---|
| CPU (NumPy) | 139.47 ms | 1x |
| GPU (custom CUDA kernel) | 4.78 ms | **29.2x** |

### 2. Matrix multiply: NumPy (CPU) vs PyTorch / cuBLAS (GPU)

| Matrix size | CPU (ms) | GPU (ms) | Speedup | GPU TFLOPS |
|---|---|---|---|---|
| 256 x 256 | 0.35 | 0.056 | 6.4x | 0.60 |
| 1024 x 1024 | 17.25 | 0.830 | 20.8x | 2.59 |
| 2048 x 2048 | 149.84 | 6.032 | 24.8x | 2.85 |
| 4096 x 4096 | 1726.09 | 32.354 | **53.3x** | **4.25** |

### 3. Neural-network inference vs batch size

The model is a small MLP: 28 inputs (matching the 28 sensor channels of the McMaster exoskeleton), two hidden layers of 256, and 4 outputs. "GPU incl. copy" counts the time to move data to the GPU and the result back. That's the realistic number.

| Batch size | CPU (ms) | GPU (ms) | GPU incl. copy (ms) | Faster |
|---|---|---|---|---|
| 1 | 0.054 | 0.095 | 0.152 | CPU |
| 8 | 0.079 | 0.179 | 0.161 | CPU |
| 64 | 0.189 | 0.110 | 0.164 | GPU |
| 512 | 0.949 | 0.109 | 0.181 | GPU |
| 4096 | 7.010 | 0.285 | 0.502 | GPU |

**The GPU only wins from batch size 64 onward.**

## What I learned

- **GPUs win when there's enough parallel work.** The matrix-multiply speedup grows from 6x to 53x as the matrices get bigger, because larger problems keep more of the GPU's cores busy.
- **Data transfer is the hidden cost.** For one prediction at a time (batch size 1), the CPU is about 3x faster than the GPU once copying data to and from the GPU is included. For a real-time system that predicts one sample at a time, a GPU isn't automatically the right choice. You have to measure.
- **Benchmarking GPUs takes care.** GPU work runs asynchronously, so the script warms up first and calls `torch.cuda.synchronize()` before stopping the timer. Otherwise the timings would be wrong.

## Run it yourself

1. Open [Google Colab](https://colab.research.google.com) and choose **Runtime → Change runtime type → T4 GPU**.
2. Paste `gpu_benchmark.py` into a cell and run it, or upload the file and run `!python gpu_benchmark.py`.

Requirements: Python 3, NumPy, PyTorch with CUDA, Numba. All of these come preinstalled on Colab.

Your numbers will vary a little between runs and between GPUs.
