# Parameter Golf — Top 5 Techniques (Simple Breakdown)

## The 5 Bullets

1. **Sliding window evaluation** — advancing the scoring window by only 64 tokens at each step (vs. full context jumps) gave −0.032 BPB at zero training cost, the largest free lunch in the competition.
2. **Int6 per-row quantization + zstd-22 + MLP 3x expansion** — int6 compression freed ~4MB that funded a 3x wider MLP, yielding ~−0.02 BPB and becoming the foundation of nearly all competitive submissions.
3. **FP16 tied embeddings** — keeping the shared embedding matrix in full precision while quantizing everything else eliminated the dual-path quantization penalty (~−0.006 BPB, near-zero cost).
4. **Legal score-first TTT** — evaluating with `torch.inference_mode()` before any weight update using AdamW (lr=0.0001, 3 epochs, frozen embeddings) delivered −0.010 to −0.020 BPB with no information leakage.
5. **Value Residual (ResFormer-style)** — caching the V matrix from layer 0 and blending it into deeper layers via two learned scalars achieved −0.015 BPB with only 22 additional parameters.

---

## Technical Explanations

### 1. Sliding Window Evaluation

Standard LM evaluation with fixed context windows means most tokens are scored with negligible prior context — at a 2048-token context on a 131K-token validation set, only ~10% of positions have meaningful context. Sliding window evaluation (stride=64) ensures every token is scored with 960+ tokens of overlap, which converts an evaluation artifact (the pre-quantization model's unchanged forward pass) into a reliable quality signal. The key insight is that the pre-quantization model already captures long-range dependencies well; you're just measuring it more faithfully. The cost is 4x evaluation time (~70s vs ~16s), but this is well within a 10-minute budget. This is a measurement methodology change with zero training implications.

---

### 2. Int6 Per-Row Quantization + MLP 3x Expansion

Int8 quantization typically saves ~2 bytes per parameter; int6 (64 levels) with per-row quantization (scale per output channel) and zstd-22 compression saves ~3 bytes per parameter. On a 21.8M-parameter model, this difference (~4MB) funds expanding MLP hidden dimension from 2x to 3x model dimension (~4M extra params). Per-row vs per-tensor quantization matters because weight distributions in transformers are highly channel-dependent — clipping at row-maximum leaves significant on-tail mass unexploited. The 3x expansion is not a free lunch: it requires either QAT (quantization-aware training via straight-through estimation) or a minimal quant gap to avoid quality regression. The stack (int6 + zstd-22 + MLP 3x) became the de facto foundation of 80%+ of competitive submissions because the size-quality tradeoff sits on the Pareto frontier in a way that int8+zlib or int5 does not.

---

### 3. FP16 Tied Embeddings

In tied embedding architectures, input token encoding and output logit projection share `W_embed`. When `W_embed` is quantized to int8, both the forward pass (input projection) and the output projection (logits ← `W_embed^T @ hidden`) suffer quantization error simultaneously, compounding through the softmax. Keeping `W_embed` in fp16 while quantizing all other tensors is a surgical exception: the ~500KB overhead is offset by shrinking the MLP slightly, but the quantization penalty drops from ~0.007 BPB to ~0.0005 BPB — a 14x reduction in roundtrip error. This generalizes to any shared-weight LM being post-training quantized: identify the weight with the highest effective "usage frequency" in the computation graph and preserve it at fp16.

---

### 4. Legal Score-First TTT (Test-Time Training)

Score-first TTT operates on the validation set after training converges. The critical constraint: every token must be scored under `torch.inference_mode()` (no gradient tracking) *before* any weight update — otherwise the model sees its own outputs during adaptation, which is information leakage (the "adapt-first" variant is invalid). The practical pipeline: process validation in ~131K-token document chunks, unfreeze the final 2 blocks, run AdamW (lr=0.0001, 3 epochs), keep embeddings frozen to prevent catastrophic forgetting. The empirical finding that AdamW >> SGD for TTT reflects that AdamW's adaptive learning rates handle the small, noisy gradient signals from single-pass document adaptation better than SGD's fixed schedule. The gain (−0.010 to −0.020 BPB) is substantial relative to most architectural tricks, suggesting that the post-training distribution shift from fineweb-1b to the validation distribution is meaningful.

---

### 5. Value Residual (ResFormer-style)

In standard multi-head attention, the value vector `V` at layer `l` is projected from the hidden state and then used in the attention computation, but information from earlier layers gets progressively filtered through Q/K/V projections at each layer. ResFormer-style Value Residual caches `V₀` (the value from layer 0) and blends it into every subsequent layer: `v_final = λ₀ · V₀ + λ₁ · V_l`, where `λ₀, λ₁` are learned scalars (2 parameters total, not per head). This creates a direct gradient pathway from early-layer value information to deep-layer predictions without adding significant parameter count or computation. The mechanism is similar to residual connections, but applied selectively to the value stream rather than the hidden state. With only 22 additional parameters, it achieves −0.015 BPB, making it one of the highest BPB-per-parameter techniques in the competition. The independently replicated results (5+ submissions) confirm the signal is genuine.
