# High-Precision O(N) Causal Linear Attention Architecture

**Author:** Subhajit Kar (SK Sir)  
**Domain:** Scalable LLMs, Hardware Acceleration & Green AI  

---

## 📌 Executive Summary
Standard Multi-Head Attention in Transformers scales quadratically $O(N^2)$, creating severe compute and VRAM bottlenecks for long-context sequences. Existing linear approximations often suffer a heavy precision drop (~70-76% similarity to Softmax).

This project introduces a **Causal $O(N)$ Linear Attention Layer** using **Learnable High-Rank MLP Projections** and associative prefix sums (`cumsum`). The architecture bridges the precision gap to **98.6% Cosine Similarity** relative to Softmax while delivering true linear scaling at $O(N)$ complexity.

To eliminate High-Bandwidth Memory (HBM) latency, we implemented a custom fused **Triton/CUDA GPU Kernel** that executes state accumulation directly inside ultra-fast GPU SRAM registers.

---

## ⚡ Key Architectural Innovations
* **Precision Bridge:** Dynamic non-linear feature projections increase representation rank, retaining 98.6% Softmax expressivity.
* **Causal Prefix Sums:** Associative cumulative operations replace $O(N^2)$ attention matrices with $O(1)$ state updates per token during inference.
* **Fused Hardware Acceleration:** Custom Triton kernel bypasses intermediate HBM reads/writes for maximum hardware efficiency.
* **Green AI Efficiency:** Drastically reduces VRAM footprint and datacenter power draw for LLM pre-training and serving.

---

## 📊 Hardware Benchmarks (NVIDIA T4 GPU Verification)

### 1. PyTorch Baseline vs Triton GPU Kernel
Verified real-world throughput across sequence lengths up to 32,768 tokens:

| Context Length ($N$) | PyTorch Latency | Triton Kernel Latency | Triton Speedup |
| :---: | :---: | :---: | :---: |
| **16,384** | `62.93 ms` | **`17.46 ms`** | **`3.60x FASTER`** 🚀 |

---

### 2. Extreme Long Context Stress Test (128K Tokens)
Stress-testing the Custom Fused Triton Kernel up to **131,072 Tokens** proves sub-linear memory growth:

| Sequence Length ($N$) | Triton Execution Latency | VRAM Allocated | Status |
| :---: | :---: | :---: | :---: |
| **32,768 Tokens** | `46.03 ms` | `73.12 MB` | PASSED ✅ |
| **65,536 Tokens** | `54.51 ms` | `113.12 MB` | PASSED ✅ |
| **131,072 Tokens (128K)** | **`108.78 ms`** | **`185.12 MB`** | **PASSED (Flat $O(N)$ Memory)** ✅ |

> 💡 **Key Takeaway:** While standard Softmax Attention explodes in VRAM and crashes at 128K tokens, our Triton Kernel completes full sequence execution in **108 ms** using under **186 MB of VRAM**.

---

## 📁 Repository Structure
* `Linear_Attention_Project.py`: Full PyTorch implementation, 4-layer Transformer stack, and synthetic training pipeline.
* `triton_kernel.py`: Custom fused Triton/CUDA C++ kernel implementation for GPU SRAM execution.
* `benchmark.py`: GPU VRAM and Latency evaluation suite.
* `Linear_Attention_Research_Paper.pdf`: Technical research report, mathematical proofs, and benchmarking analysis.
*
