# FlashAttention CUDA Implementation

CUDA-based implementation of FlashAttention with optimized shared memory tiling and online softmax for efficient transformer attention computation.

Refer to the project report PDF for detailed implementation details, algorithm explanations, performance analysis, and experimental results.

## Overview

This project compares three implementations of the attention mechanism:

- CPU Attention Baseline
- Naive CUDA Global Memory Attention
- Optimized FlashAttention-style Shared Memory Kernel

The goal is to reduce memory bottlenecks caused by standard attention implementations by leveraging:

- Shared Memory (SRAM)
- Tiling
- Online Softmax
- CUDA Parallelism

---

## Features

- Exact attention computation
- CUDA shared memory optimization
- Online softmax for numerical stability
- Benchmarking against CPU and naive GPU implementations
- CMake build system
- Dataset generation utilities
- Triton reference implementation

---

## Repository Structure

```bash
FlashAttention/
├── src/
│   ├── fa_cpu/           # CPU attention implementation
│   ├── fa_gpu/           # CUDA implementations
│   ├── triton/           # Triton reference implementation
│   ├── datasets/         # Generated Q, K, V inputs
│   └── CMakeLists.txt
└── README.md
└── fa_benchmar.sh       # nsight compute
└── run_tests.sh         # test cases for random N and d values


```

---

## Build

```bash
cmake -B build
cmake --build build
bash run_test.sh
```

---

## NSight Compute

Additionally, run the benchmark script to get the nsight compute file.

---

## Performance Results

| Implementation    | Execution Time |
| ----------------- | -------------- |
| CPU Attention     | 5831 ms        |
| GPU Global Memory | 175 ms         |
| GPU Shared Memory | 18 ms          |

### Speedups

- Global GPU vs CPU: **33×**
- Shared Memory vs CPU: **322×**
- Shared Memory vs Global GPU: **9.7×**

---

## Standard Attention

Standard attention materializes the full attention matrix in memory:

```math
S = QK^T
```

```math
P = softmax(S)
```

```math
O = PV
```

This leads to:

- High HBM memory traffic
- \(O(N^2)\) memory complexity
- Poor scalability for large sequence lengths

---

## FlashAttention Optimization

Our implementation improves efficiency using:

- Block-wise tiling
- Shared memory reuse
- Online softmax computation
- Reduced global memory access

### Key Idea

Instead of storing the full attention matrix in HBM, FlashAttention computes attention in tiles directly on-chip using SRAM.

---

## Procedure

1. Generate Q, K, and V datasets
2. Run CPU reference implementation
3. Run naive GPU implementation
4. Run shared memory FlashAttention kernel
5. Compare outputs for correctness
6. Benchmark runtime performance

---

## Challenges

- Numerical stability during softmax
- CUDA shared memory management
- Floating-point precision differences
- CMake configuration and integration

---

## Limitations

- Forward pass only
- Limited benchmarking scope
- Missing advanced kernel fusion optimizations
- Hardware-dependent performance

---

## References

1. Vaswani et al., _Attention Is All You Need_
2. Dao et al., _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness_

---

## Team

- Apar Pokhrel
- Che Kwanga
- Pisit Nakhonekhong
- Jose Lopez
- Kopil Sharma
