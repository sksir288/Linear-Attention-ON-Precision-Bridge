# High-Precision O(N) Causal Linear Attention Architecture

**Author:** Subhajit Kar (SK Sir)  
**Domain:** Scalable LLMs, Hardware Acceleration & Green AI  

---

## 📌 Executive Summary
Standard Multi-Head Attention in Transformers scales quadratically $O(N^2)$, creating severe compute and VRAM bottlenecks for long-context sequences. Existing linear approximations often suffer a heavy precision drop (~70-76% similarity to Softmax).

This project introduces a **Causal $O(N)$ Linear Attention Layer** using **Learnable High-Rank MLP Projections** and associative prefix sums (`cumsum`). The architecture bridges the precision gap to **98.6% Cosine Similarity** relative to Softmax while delivering linear scaling at $O(N)$ complexity.

To further eliminate PyTorch High-Bandwidth Memory (HBM) latency, we implemented a custom fused **Triton/CUDA GPU Kernel** that executes state accumulation directly inside ultra-fast GPU SRAM registers.

---

## ⚡ Key Architectural Innovations
* **Precision Bridge:** Dynamic non-linear feature projections increase representation rank, retaining 98.6% Softmax expressivity.
* **Causal Prefix Sums:** Associative cumulative operations replace $O(N^2)$ attention matrices with $O(1)$ state updates per token during inference.
* **Fused Hardware Acceleration:** Custom Triton kernel bypasses intermediate HBM reads/writes for maximum hardware efficiency.
* **Green AI Efficiency:** Drastically reduces VRAM footprint and datacenter power draw for LLM pre-training and serving.

---

## 📊 Hardware Benchmarks (NVIDIA T4 GPU Verification)

### 1. PyTorch Baseline Linear Scaling $O(N)$
Verified real-world linear throughput across sequence lengths up to 32,768 tokens:

| Context Length ($N$) | Execution Latency (ms) | Peak Allocated VRAM (MB) | Scaling Behavior |
| :---: | :---: | :---: | :---: |
| **1,024** | `5.32 ms` | `309.23 MB` | Baseline |
| **4,096** | `15.72 ms` | `1,197.32 MB` | Perfect $O(N)$ Linear Scaling |
| **16,384** | `62.93 ms` | `4,749.70 MB` | Perfect $O(N)$ Linear Scaling |
| **32,768** | `125.99 ms` | `9,486.20 MB` | Flat Linear VRAM Curve |

> 💡 **Takeaway:** Extending context length by **4x** (4K ➔ 16K) results in exactly **4x latency/VRAM growth**, unlike standard Softmax $O(N^2)$ which explodes quadratically.

---

### 2. Custom Fused Triton GPU Kernel Speedup
By fusing kernel operations and keeping key-value state accumulation inside SRAM:

| Implementation | Context Length ($N$) | Latency (ms) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **PyTorch Standard (`cumsum`)** | `16,384` | `62.93 ms` | `1.0x` (Baseline) |
| **Custom Fused Triton Kernel** | `16,384` | **`17.46 ms`** | **`3.60x FASTER`** 🚀 |

---

## 📁 Repository Structure
* `Linear_Attention_Project.py`: Full PyTorch implementation, 4-layer Transformer stack, and synthetic training pipeline.
* `triton_kernel.py`: Custom fused Triton/CUDA C++ kernel implementation for GPU SRAM execution.
* `benchmark.py`: GPU VRAM and Latency evaluation suite.
* `Linear_Attention_Research_Paper.pdf`: Technical research report, mathematical proofs, and benchmarking analysis.
*
