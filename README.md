# Linear-Attention-ON-Precision-Bridge# High-Precision O(N) Causal Linear Attention Architecture

**Author:** Subhajit Kar (SK Sir)  
**Domain:** Scalable LLMs, Efficient Transformers & Green AI  

## Key Achievements
- **Precision Bridge:** Retains **98.6% Cosine Similarity** to O(N²) Softmax using Learnable MLP Projections.
- **Linear Scaling:** **~80x execution speedup** at N = 16,384 sequence length.
- **O(1) Inference:** Associative prefix sum (`cumsum`) updates for constant-time token generation.

## Repository Structure
- `Linear_Attention_Project.py`: Full model architecture, 4-layer GPT stack, and training pipeline.
- `Linear_Attention_Research_Paper.pdf`: Complete research report, benchmarks, and mathematical proofs.
-
