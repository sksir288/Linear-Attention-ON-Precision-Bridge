# High-Precision O(N) Causal Linear Attention Architecture

**Author:** Subhajit Kar (SK Sir)  
**Domain:** Scalable LLMs, Efficient Transformers & Green AI  

---

## 📌 Executive Summary
Standard Multi-Head Attention in Transformers scales quadratically $O(N^2)$, creating severe compute and VRAM bottlenecks. Existing linear approximations often suffer a heavy precision drop (~70-76% similarity to Softmax).

This research introduces a **Causal $O(N)$ Linear Attention Layer** using **Learnable High-Rank MLP Projections** and associative prefix sums (`cumsum`). The architecture bridges the precision gap to **98.6% Cosine Similarity** relative to Softmax while delivering an **~80x execution speedup** at $N = 16,384$.

---

## ⚡ Key Architectural Innovations
* **Precision Bridge:** Dynamic non-linear feature projections increase rank, retaining 98.6% Softmax expressivity.
* **Causal Prefix Sums:** Associative cumulative operations replace $O(N^2)$ attention matrices with $O(1)$ state updates per token during inference.
* **Green AI Optimization:** Significantly reduces GPU memory overhead, power draw, and datacenter energy footprint.

---

## 📊 Hardware Benchmarks (NVIDIA T4 GPU Verification)

Verified real-world linear throughput across sequence lengths up to 32,768 tokens:

| Context Length ($N$) | Execution Latency (ms) | Peak Allocated VRAM (MB) | Scaling Behavior |
| :---: | :---: | :---: | :---: |
| **1,024** | `5.32 ms` | `309.23 MB` | Baseline |
| **4,096** | `15.72 ms` | `1,197.32 MB` | Perfect $O(N)$ Scaling |
| **16,384** | `62.93 ms` | `4,749.70 MB` | Perfect $O(N)$ Scaling |
| **32,768** | `125.99 ms` | `9,486.20 MB` | Flat Linear VRAM Curve |

> 💡 **Key Takeaway:** Extending sequence length by **4x** (4K ➔ 16K) results in exactly **4x latency and memory growth**, proving true $O(N)$ complexity unlike standard Softmax $O(N^2)$ which explodes exponentially.

---

## 📁 Repository Contents
* `Linear_Attention_Project.py`: Full PyTorch implementation, 4-layer GPT stack, and training pipeline.
* `benchmark.py`: GPU VRAM and Latency benchmarking suite.
* `Linear_Attention_Research_Paper.pdf`: Official research report, mathematical proofs, and benchmarking analysis.
*
