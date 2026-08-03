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

## 📊 Empirical Benchmarks (Micro-GPT Training)
| Iteration Step | Training Loss | Optimization State |
| :---: | :---: | :---: |
| **Step 1** | `3.8602` | Initial Convergence |
| **Step 50** | `1.7579` | Rapid Gradient Descent |
| **Step 100** | `1.4714` | Pattern Alignment |
| **Step 150** | `0.9732` | High Precision State |
| **Step 200** | `0.5327` | Verified Convergence |

---

## 📁 Repository Contents
* `Linear_Attention_Project.py`: Full PyTorch implementation, 4-layer GPT stack, and training pipeline.
* `Linear_Attention_Research_Paper.pdf`: Official research report, mathematical proofs, and benchmarking analysis.
*
