# Parameter Golf Techniques Audit — Deep Research Report
*Research date: 2026-03-24 | Covers github.com/openai/parameter-golf through 560+ PRs, 21 record entries, 11 issues*

---

## LEVEL 1: Executive Key Insights

### SMALLER — Techniques That Improve Quality Under Size Constraints

**1. Sliding window evaluation is the single largest free lunch in the competition (−0.032 BPB, zero training cost).** By advancing a fixed scoring window by only 64 tokens at each step instead of jumping by the full context length, every token receives 960+ tokens of prior context instead of near-zero. This converts an evaluation artifact into a reliable quality signal — the pre-quantization model unchanged, yet measured BPB drops from 1.2244 to 1.1925 at baseline. Evaluation time grows 4x, but the 10-minute eval budget makes this affordable. **Transfers directly to any autoregressive LM evaluation where the validation set is longer than the context window.** [Source: Records/2026-03-19_SlidingWindowEval, PR #60]

**2. MLP 3x expansion funded by int6 quantization is the most reliable single architecture gain (~−0.02 BPB).** Widening the hidden dimension from 2x to 3x the model dimension adds ~4M parameters that, at int6 precision with zstd-22, still fit within the 16MB budget. The size saved by int6 over int8 (roughly 4MB on a 9-layer model) directly funds the wider MLP. This is not a free lunch — it requires int6 QAT or a minimal quant gap — but it is among the most validated, independently confirmed techniques in the competition. **Transfers to any scenario where you can trade lower per-weight precision for more weights.** [Source: PR #70, Records/2026-03-19_MixedQuant, 50+ independent reproductions]

**3. FP16 tied embeddings (preserving the embedding from int8) eliminates the dual-path quantization penalty (~−0.006 BPB, near-zero cost).** When input and output share the same embedding matrix, int8 quantization degrades both token encoding and logit projection simultaneously, compounding error. Keeping this one tensor in fp16 and slightly shrinking the MLP to compensate adds a trivial ~500KB overhead but cuts the quantization roundtrip penalty from ~0.007 to ~0.0005 BPB. **Transfers to any shared-embedding LM being post-training quantized.** [Source: Records/2026-03-18_FP16Embed_WD3600, PR #60]

**4. Per-row int6 quantization with zstd-22 compression strictly dominates int8+zlib for the same artifact budget.** The combination of 6-bit integer quantization (64 levels) with zstd-22 achieves 4–5x better size reduction than int8+zlib on typical weight distributions. With QAT training to close the roundtrip gap, the size-versus-quality Pareto frontier sits at int6, not int8. Int4 has a catastrophic +0.065 BPB gap; int5 MLP works for undertrained models but shows +0.007 BPB gap in the competition context. **The int5/int6 boundary appears to be a general sweet spot for LM weight quantization.** [Source: PR #480, PR #238, Records/2026-03-19_MixedQuant]

**5. Mixed-precision quantization by layer type squeezes additional capacity from the budget at nearly zero quality cost.** Using int5 for MLP weights (less sensitivity) and int6 for attention weights (more sensitivity) achieves ~1–2MB additional headroom over uniform int6, which can be reinvested in more layers or wider MLPs. The technique was independently validated by at least 4 top-performing submissions. Uniform int5 across all layers is risky — it amplifies error through undertrained weights — but per-type mixing is robust. **Directly transferable to any mixed-precision quantization pipeline.** [Source: Records/2026-03-20_10L_Int5MLP, PR #466, PR #544]

**6. GPTQ-lite (5 clip-percentile search per weight row, pick min-MSE) closes the post-training quantization gap at zero training cost (~−0.0006 BPB vs round-max clipping, −0.0024 BPB for full GPTQ).** Standard quantization clips at row-max, which is dominated by outliers. Searching only 5 percentile thresholds (100%, 99.9%, 99.5%, 99%, 98%) and selecting the one minimizing reconstruction MSE adds <5 seconds of compute and systematically beats naive clipping. Full GPTQ with Hessian calibration on 256 samples does better but requires more time. **This is the cheapest quantization improvement that works universally.** [Source: Record/2026-03-22_GPTQ-lite, PR #379, PR #528]

**7. Codebook quantization + Huffman entropy coding beats zstd on the weight distribution non-uniformity (~21% additional compression over int6+zstd).** Standard zstd is a general-purpose compressor. LM weight indices after quantization have non-uniform distributions that a custom Huffman coder exploits directly. A three-stage pipeline (K-means codebook per tensor type → Huffman coding → zstd-22) achieved 14.12MB for a 27M-parameter model that int6+zstd would push to 18+MB, opening ~1.88MB of additional model capacity. **Applies to any scenario where you need to maximize model size within a hard byte budget.** [Source: PR #532 by NotADevIAmaMeatPopsicle]

**8. BigramHash embedding (hash table over consecutive token pairs) contributes ~−0.01 BPB at tiny parameter cost.** Mapping consecutive token pairs through a 4096–10240 bucket hash table into a 128-dimensional embedding that projects into the model dimension provides n-gram statistics that the vocabulary-1024 tokenizer loses. Larger buckets (up to 10240) consistently improve BPB with diminishing returns around 8192–10240. TrigramHash (three-token context) provides marginal additional gain. **Transferable to any small-vocabulary model where higher-order token statistics matter.** [Source: PR #164, Records/2026-03-20_Int6_MLP3x_SmearGate_BigramHash, PR #486]

---

### FASTER — Techniques That Maximize Training Efficiency Under Time Constraints

**9. FlashAttention 3 on Hopper GPUs (H100) provides 15–20% more training steps in the same wall-clock budget, directly translating to ~0.003–0.005 BPB gain.** FA3 is specifically optimized for the H100 Tensor Core architecture. Switching from standard attention or FA2 to FA3 reduces the per-step time by 15–20%, allowing ~1,000 more gradient steps within 10 minutes. This is a pure infrastructure gain with no quality trade-off. **Any training pipeline targeting H100s should use FA3; the gain is free.** [Source: PR #375, PR #164, multiple records]

**10. EMA (decay=0.997, every step) outperforms SWA by ~0.003 BPB and has lower implementation overhead.** Standard SWA collects checkpoints at fixed intervals, which creates periodic CPU-GPU transfers and can miss better-converged states. EMA maintains a running average with minimal overhead and achieves smoother weight trajectories. Critically: a naive EMA implementation that clones the full state dict to CPU every step introduces a 32% throughput penalty. The correct implementation tracks only the relevant tensors. EMA that stacks on top of tight SWA (final 600 steps) provides additional benefit. **EMA is now considered the default averaging strategy for constrained training.** [Source: PR #287, PR #360, PR #375]

**11. At 86ms/step, every 1ms of per-step overhead costs ~0.006 BPB — techniques must be evaluated by their step-time cost, not just quality contribution.** This is the meta-insight that explains most technique failures in this competition. MTP failed not because multi-token prediction is wrong but because it added 28ms/step, costing more in lost training than it recovered in gradient signal. EMA with naive CPU clone cost 32% of steps. MoE routing overhead consumed 12ms/step. The throughput-adjusted BPB delta is the correct figure of merit. **Any constrained training scenario should calculate: (quality_gain_per_token × tokens_lost_from_overhead) before adopting a technique.** [Source: PR #375 systematic analysis]

**12. Parallel Muon with parameter banking reduces per-step communication overhead by ~3% via batched Newton-Schulz orthogonalization.** Consolidating 66 separate weight matrices into four 3D tensor banks enables a single batched `torch.bmm` call for all orthogonalization, reducing kernel launch overhead. Manual all-reduce scheduling (bypassing DDP for bank parameters) overlaps communication with compute. The gain is small in absolute terms but adds ~200 steps over 10 minutes. **The pattern generalizes: batching small optimizer operations improves throughput on large clusters.** [Source: PR #399]

**13. Extending warmdown beyond the actual training steps ("over-scheduling") produces tighter weight distributions that compress better.** Setting WARMDOWN_ITERS far beyond the training-step count forces the learning rate onto a steeper always-decaying trajectory. This produces weight distributions with fewer outliers, reducing the quantization penalty by ~0.010 BPB. The mechanism is that lower final learning rates create more "compressible" weights. **Transferable to any model where post-training quantization quality matters and the training budget is fixed.** [Source: Records/2026-03-19_WarmdownQuantization, PR #374]

---

### BETTER — Techniques That Improve Model Quality Independent of Constraints

**14. Exclusive Self Attention (XSA) applied to the final 3–4 layers adds ~−0.005 BPB with zero parameters.** XSA subtracts the component of each attention output aligned with that token's own value vector ("removing self-value bias"), forcing the model to capture information it doesn't already know. The efficient GQA-aware implementation uses reshape+broadcast instead of `repeat_interleave`, reducing overhead from 7ms to 2ms per step. Applied selectively to only the final layers (where self-bias is highest) is better than applying uniformly. **XSA + TTT is a negative combination; they target overlapping capabilities.** [Source: Records/2026-03-20_11L_XSA4_EMA, PR #287, PR #303]

**15. Partial RoPE (apply rotary embeddings to only ~25% of head dimensions) improves generalization for free.** The remaining 75% of dimensions attend without positional bias, learning position-invariant patterns. This zero-parameter change yielded a consistent −0.002 BPB improvement in the record stack. The mechanism: position-invariant attention heads can capture semantic similarity across document positions, complementing the position-aware heads. **Directly transferable; a drop-in modification to any RoPE-based transformer.** [Source: Records/2026-03-21_PartialRoPE, PR #315]

**16. Value Residual (ResFormer-style) delivers −0.015 BPB improvement with only 22 additional parameters.** Caching the V matrix from layer 0 and blending it into subsequent layers via two learned scalar weights (`v = λ₀·v₀ + λ₁·v`) provides a direct pathway for early-layer value information to influence deep layers. This was independently validated by 5+ community submissions. **Exceptionally parameter-efficient; the technique has among the highest BPB/parameter ratios of any technique found.** [Source: PR #487 ablation, PR #490, PR #507]

**17. Legal score-first TTT provides −0.010 to −0.020 BPB with no training changes.** Score-first TTT evaluates every token under `torch.inference_mode()` before any weight update, preserving evaluation integrity. Processing the validation set in 131K-token chunks with AdamW (lr=0.0001, 3 epochs, freeze all but the final 2 blocks) provides substantial gains. AdamW significantly outperforms SGD for TTT. Freezing embeddings is critical — allowing embedding updates causes catastrophic forgetting. **The causal/document-isolation constraint is non-negotiable; adapt-first TTT is information leakage.** [Source: PR #528, PR #473, Issue #402]

**18. Weight decay on Muon directly controls compressed artifact size at a rate of ~1.5–2MB per 0.01 WD unit.** Higher weight decay (0.04 vs 0.01) produces smaller parameter magnitudes, which compress more efficiently under zstd. This effect is separate from generalization — it enables fitting larger models within the size cap. WD=0.04 became the community default after PR #198. Setting WD=0 for the TTT phase prevents forgetting. **In any size-constrained scenario with learned compression, weight decay is a knob for trading generalization regularization against artifact size.** [Source: PR #375, Records/2026-03-20_Int6_MLP3x_SmearGate_BigramHash]

**19. LN Scale (divide RMSNorm output by √(layer_idx+1)) stabilizes deep model training at zero cost.** This depth-dependent dampening prevents the contribution of deeper layers from dominating the residual stream, stabilizing gradient flow and improving convergence in 11-layer models. Combined with Partial RoPE in PR #315, these two zero-parameter tricks yielded −0.0023 BPB. **Likely underexplored in the broader community; depth-dependent normalization is a free regularizer.** [Source: Records/2026-03-21_PartialRoPE, PR #315]

**20. SWA that exploits the quantization domain reversal is qualitatively different from standard SWA.** With sufficient checkpoints (84+) averaged, SWA-smoothed weights can exhibit lower BPB after int6+zstd quantization than before — a reversal of the typical quantization penalty. The mechanism: checkpoint averaging eliminates quantization-sensitive parameter outliers. Weight entropy regularization (adding `loss += λ * weight_entropy` during training) amplifies this effect by encouraging consistent distributions across checkpoints, yielding an additional −0.028 BPB. **This pattern suggests that for any heavily quantized model, the right optimization target is the post-quantization loss, not pre-quantization.** [Source: PR #238, PR #459]

---

## LEVEL 2: Technical Report

### Record Progression

#### Record #0 (Baseline): 1.2244 BPB
- **Architecture**: 9 layers, 512d, 1024 vocab, 2x MLP, tied embeddings, 4 KV heads
- **Quantization**: Int8 + zlib
- **Optimizer**: Adam (default)
- **Reached step 13,780/20,000** before 10-min cap; 43ms/step
- **Model size**: 15.86MB compressed

#### Record #1 (2026-03-17): 1.1928 BPB — LoRA TTT
- **Key techniques**: Per-document LoRA rank-8 TTT at eval, document isolation, strided evaluation
- **Ablation**: Document isolation alone −0.011 BPB; strided eval alone −0.034 BPB; TTT alone −0.037 BPB
- **Transferability**: HIGH — Per-document score-first TTT with LoRA adapters is broadly applicable. The surprising finding: most gain comes from document isolation and strided eval, not TTT itself.

#### Record #2 (2026-03-18): ~1.2197 BPB — FP16 Embed + Extended Warmdown
- **Delta**: −0.005 BPB
- **Technique**: FP16 tied embedding preservation; WARMDOWN_ITERS=3600; MATRIX_LR raised to 0.06
- **Failed approaches documented**: SwiGLU, depth recurrence, QAT, lzma — all counterproductive in this period
- **Transferability**: HIGH — FP16 embedding preservation is a standard practice in deployment now

#### Record #3 (2026-03-18): 1.2058 BPB — Long Context Seq2048
- **Delta**: −0.018 BPB
- **Technique**: 2048-token sequences; NTK-RoPE extrapolation; differential LR for embeddings vs matrices
- **Transferability**: HIGH — longer context is universally beneficial; the NTK-RoPE trick for extrapolation is widely used

#### Record #4 (2026-03-19): 1.2014 BPB — Training Opt Seq4096
- **Delta**: −0.004 BPB
- **Technique**: 4096-token sequences; tuned Muon (momentum 0.99 vs 0.95, reduced LR, smaller batch for more updates); extended warmdown 3000 steps
- **Transferability**: HIGH — Muon momentum 0.99 became the community default

#### Record #5 (2026-03-19): 1.1925 BPB — Sliding Window Eval
- **Delta**: −0.009 BPB from prior best; −0.032 BPB over naive baseline
- **Technique**: Stride-64 overlapping evaluation windows; each token scored with 960+ prior-context tokens; evaluation time 70s (up from 16s)
- **Transferability**: HIGH — largest single evaluation-methodology gain; universally applicable to LM evaluation

#### Record #6 (2026-03-19): 1.1630 BPB — Mixed Quant Int6 + MLP3x + Sliding Window
- **Delta**: −0.030 BPB
- **Technique**: Per-row int6 quantization on MLP/attention; int8 on embeddings; zstd-22; MLP 3x expansion (21.8M params); seq1024 for speed; 12,395 steps
- **Key insight**: Mixing int6 body with int8 embeddings reduced quantization penalty from +0.048 to +0.0015 BPB
- **Transferability**: HIGH — this combination became the foundation stack for 80% of subsequent competitive submissions

#### Record #7 (2026-03-19): 1.1598 BPB — Seq2048 + FP16Emb + QAT
- **Delta**: −0.003 BPB
- **Technique**: STE int6 QAT (zero quant gap); MLP 2.6x; FP16 tied embedding; seq2048; zstd-22
- **Transferability**: HIGH — STE QAT eliminating the quant roundtrip gap is broadly applicable

#### Record #8 (2026-03-19): 1.1556 BPB — SmearGate + OrthoInit + Muon WD
- **Delta**: −0.004 BPB
- **Technique**: SmearGate (learned gate blending prev token, ~512 params); OrthoInit + muP scaling; Muon with decoupled WD=0.01; int6 QAT; MLP 3x; U-Net skips; sliding window
- **Transferability**: HIGH — OrthoInit is broadly beneficial; SmearGate is novel but simple

#### Record #9 (2026-03-19): ~1.1500 BPB — 11L Int6 QAT + MLP3x (stack)
- **Delta**: −0.005 BPB
- **Technique**: 11 layers (depth increase from 9 to 11); relu² activation; 3x MLP; int6 QAT; 26.5M parameters
- **Transferability**: HIGH — depth + quantization interaction (int6 headroom funds depth)

#### Record #10 (2026-03-20): 1.1458 BPB — Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA
- **Delta**: −0.029 BPB from prior
- **Technique**: Seven techniques stacked; BigramHash (4096 buckets, dim=128); SWA over final 50% of training; WD=0.04; 22M params at 15.86MB
- **Key**: "MLP 3x was the single largest contributor"; BigramHash brought the n-gram statistics the small vocab lost
- **Transferability**: HIGH across all seven components

#### Record #11 (2026-03-20): 1.1307 BPB — 11L + Efficient Partial XSA + FA3 + SWA
- **Delta**: −0.015 BPB
- **Technique**: XSA on final 3 layers (GQA-aware, zero alloc via reshape+broadcast); FA3 on Hopper; SWA 13 checkpoints
- **Transferability**: HIGH — XSA is zero-parameter, broadly applicable; FA3 is hardware-specific

#### Record #12 (2026-03-20): 1.1428 BPB — 10L Int5-MLP + BigramHash(10240) + SWA
- **Delta**: (different track from #11; this is the official leaderboard entry at 1.1428)
- **Technique**: Mixed int5 MLP / int6 attention / fp16 embed; BigramHash 10240 buckets; tight SWA (last 40% of warmdown, 24 checkpoints every 50 steps)
- **Ablation**: int5 MLP −0.003; WD tuning −0.0001; SWA tuning −0.0006; BigramHash 8192→10240 −0.0008
- **Transferability**: HIGH — the selective SWA window (collect only from later warmdown) is a general improvement

#### Record #13 (2026-03-20): 1.1271 BPB — 11L XSA4 + EMA + Int6 MLP3x + WD=0.04
- **Delta**: −0.005 BPB from prior 11L record
- **Techniques introduced**: XSA on final 4 layers (~0.002 BPB); EMA decay=0.997 replacing SWA (~0.003 BPB)
- **Transferability**: HIGH — EMA > SWA is well-validated here; XSA is zero-param

#### Record #14 (2026-03-21): 1.1248 BPB — 11L + Partial RoPE + LN Scale + EMA + XSA4
- **Delta**: −0.0023 BPB
- **Techniques introduced**: Partial RoPE 16/64 dims; LN Scale 1/sqrt(layer+1)
- **Critical bug**: Late QAT flag was dead code (torch.compile constant-fold)
- **Transferability**: HIGH — both are zero-param architectural improvements

#### Record #15 (2026-03-22, current SOTA): 1.1233 BPB — 11L EMA + GPTQ-lite + Warmdown3500 + QAT@0.15
- **Delta**: −0.0013 BPB from previous record
- **Techniques**: GPTQ-lite (5 clip percentiles, min-MSE row clipping) −0.0006; EMA stacked with tight SWA −0.0006; warmdown 3500 + late QAT threshold 0.15 −0.0003
- **Architecture**: 11L, 512d, 8h/4KV, U-Net skips, Efficient Partial XSA, Partial RoPE, SmearGate, BigramHash(2048), Shared Value Embeddings layers 9-10
- **Transferability**: HIGH — GPTQ-lite is universally applicable post-training

---

### Pull Requests

#### PRs with validated record-level results:

**PR #70** — 1.1659 BPB — [jfprincz]
- Techniques: Int6 per-row + zstd-22; MLP 3x; Sliding window eval (stride=256)
- Established the foundational "int6+MLP3x+sliding" stack used by 50+ subsequent PRs
- Transferability: HIGH

**PR #164** — 1.1524 BPB — [jfprincz]
- Added: OrthoInit + muP; BigramHash; SmearGate; Muon tuning; seq2048; FA3
- 8 techniques stacked systematically
- Transferability: HIGH

**PR #198** — 1.1318 BPB — [jfprincz]
- Added: 11 layers; WD=0.04; tight SWA; stride=64 sliding window
- Established WD=0.04 as community default
- Transferability: HIGH

**PR #287** — 1.1271 BPB — [jfprincz]
- Added: XSA on final 4 layers; EMA replacing SWA (decay=0.997)
- Transferability: HIGH

**PR #315** — 1.1248 BPB — [jfprincz]
- Added: Partial RoPE (16/64 dims); LN Scale (1/sqrt(layer+1))
- Note: Late QAT was dead code due to torch.compile constant-folding
- Transferability: HIGH

**PR #374** — 1.1246 BPB — [unnir]
- Technique: Tight SWA (restrict to scale<0.2 threshold, last ~600 steps)
- Zero SWA penalty while maintaining averaging benefit
- Transferability: HIGH

**PR #379** — 1.1257 BPB — [dannywillowliu-uchi]
- Technique: GPTQ-lite (5 clip percentiles per row, pick min MSE)
- First validated use of row-optimal clipping over max-clipping
- Transferability: HIGH

**PR #442** — 1.1027 BPB — [sjp611]
- Technique: AdamW TTT replacing SGD TTT (3-line diff); AdamW lr=0.0005, 10 epochs; −0.019 BPB vs prior TTT SOTA
- Note: Single-seed result pending multi-seed validation
- Transferability: HIGH — AdamW for TTT is a strong finding

**PR #466** — 1.1354 BPB (3-seed mean) — [simonbissonnette]
- Technique: BigramHash (12288 buckets); Mixed Int5; FA3; EMA; 11 layers
- Larger BigramHash bucket count (12288 vs 8192) provides marginal gain
- Transferability: MEDIUM — bucket count tuning is task-specific

**PR #473** — 1.1214 BPB — [abaybektursun]
- Technique: Legal score-first TTT; Parallel Muon + Parameter Banking; BigramHash 3072 buckets
- SGD for TTT was best at low epochs; freeze 2 blocks prevents forgetting at high epochs
- Transferability: HIGH

**PR #487** — ablation study — [anantdgoel]
- Technique: Value Residual (−0.015 BPB, 22 params); Gated Attention (−0.003 BPB, 37K params)
- Independently adopted by 5+ subsequent submissions
- Transferability: HIGH — these are parameter-efficient architectural improvements

**PR #490** — 1.0891 BPB — [amaljithkuttamath]
- Technique: Value Residual + Gated Attention + AdamW TTT; 14.2MB artifact
- Pending multi-seed validation
- Transferability: HIGH

**PR #507** — 1.1558 BPB — [skarakulak]
- Technique: Catalytic Residuals (learned per-dimension gates on attn+MLP outputs, initialized 1.0); Gated U-Net skips; SwiGLU 3x; Value Residual
- Catalytic Residuals as a gating mechanism on residual contributions is novel
- Transferability: MEDIUM — needs validation across tasks

**PR #528** — 1.1195 BPB — [EthanYangTW]
- Technique: GPTQ (Hessian-aware, Cholesky, 256-sample calibration, −0.0024 BPB over GPTQ-lite); Early QAT 1750 steps; Legal score-first TTT AdamW 3 epochs; 4.7M params unfrozen
- Full GPTQ beats GPTQ-lite by ~0.002 BPB but requires more compute
- Transferability: HIGH

**PR #532** — 1.0487 BPB (withdrew due to TTT violation) — [NotADevIAmaMeatPopsicle]
- Technique: Codebook + Huffman compression (K-means per tensor type, Huffman entropy coding, custom binary format); 14.12MB artifact vs 18+MB for int6+zstd
- PR closed for TTT compliance; compression pipeline itself is valid
- Transferability: HIGH for the compression pipeline; 21% better than zstd

**PR #538** — 1.1511 BPB — [cruz-andr]
- Technique: FP8 training (TransformerEngine E4M3/E5M2); Arithmetic coding (per-tensor histogram compression); Early SWA (step 4500 vs late)
- FP8 gives 1.3-1.5x throughput; Arithmetic coding ~1MB better than zstd
- Transferability: HIGH (FP8 training); MEDIUM (arithmetic coding — custom implementation required)

**PR #544** — 1.1179 BPB — [EthanYangTW]
- Technique: Int5 GPTQ + 33.6M params (3.5x MLP); 8192-bucket BigramHash; XSA all layers; score-first AdamW TTT
- Train larger model with int5 compression to fit 16MB while increasing quality
- Transferability: HIGH — "train large, quantize hard" is a general principle

#### Negative results and failure modes:

**PR #375** — Systematic negative results on PR #315 base — [charmquark1984]
- Failed: MTP (+0.028 BPB, throughput cost), INT4 (0.06 BPB gap), canonical layers (48% step overhead), memory tokens, gradient-guided quantization, cautious WD, L1 regularization, label smoothing, 1M batch, full-run QAT
- **Positive**: EMA > SWA by 0.003; WD=0.01→WD controls artifact size; 786K > 524K batch by 0.004; FA3 = +15-20% steps
- Transferability: HIGH meta-insight about throughput-adjusted evaluation

**PR #303** — XSA + EMA + TTT negative interaction — [sseanliu]
- TTT hurts by 0.016 BPB when combined with XSA+EMA; XSA and TTT both target local context modeling, creating redundancy
- Transferability: HIGH insight — architectural redundancy between inference-time and test-time adaptations

**PR #360** — QAT + EMA degraded performance (+0.018 BPB vs baseline) — [MultiFe22]
- Root cause: QAT cost 8% throughput; EMA naive CPU clone cost 32% throughput; net loss more than quality gain
- Transferability: HIGH — must calculate throughput-adjusted effect before adopting any technique

**PR #363** — Depth recurrence amplifies quantization error ~900x — [evangelinehelsinki]
- Weight-sharing across 3 recurrence cycles amplified int6 gap to +1.14 BPB; int8 gap was +0.37 BPB
- **Conclusion**: Depth recurrence requires higher-precision quantization; int6 is incompatible with standard depth recurrence
- Transferability: HIGH warning — critical architecture-quantization interaction

**PR #367** — BitNet b1.58 systematic study — [ksang123]
- XSA and weight decay cause complete training plateaus at val_loss 2.4 for ternary models
- SmearGate, BigramHash, OrthoInit provide no benefit in ternary regime
- EMA/SWA incompatible with ternary quantization
- Better path: int4 with late QAT as middle ground
- Transferability: HIGH — identifies hard constraints on techniques compatible with ternary quantization

**PR #480** — MoE negative result — [imyesung]
- 2-expert soft-routing MoE with 1.5x MLP per expert: +0.016 BPB at step 2000
- Int5 MLP: +0.007 BPB penalty; Int4 MLP: +0.065 BPB penalty
- Transferability: HIGH — MoE routing overhead at small scale is not compensated by capacity

**PR #238** — SWA reversal + Int5 failure — [kellyvv]
- With 84 checkpoints, SWA-smoothed model has LOWER post-quant BPB than pre-quant (reversal)
- Int5+int6 mixing at this training duration: +1.1 BPB gap (catastrophic for undertrained models)
- Transferability: HIGH — int5 viability depends strongly on training duration/convergence

**PR #183** — Cache LM negative on FineWeb — [anantdgoel]
- Unigram cache LM (λ=0.02) hurts by +0.002 BPB on FineWeb
- Reason: FineWeb documents are too short and diverse for cache to accumulate useful signal
- LoRA TTT positive: −0.003 BPB on same base
- Transferability: HIGH — cache LMs work on long homogeneous documents (Wikipedia, code), not web text

---

### Issues

**Issue #82** — Tips for Newbies [jordankzf]
- Infrastructure: Datacenter-locked network volumes; compile takes 3-4 min on H100 (not counted in training time); RTX Pro 6000 hangs on compilation (use Hopper or Ada Lovelace only)
- Pre-tokenized SP-4096 and SP-8192 data available on HuggingFace (sproos/parameter-golf-tokenizers)
- Key tip: Test on single GPU (2-3 min) before scaling to 8-GPU run

**Issue #402** — TTT Information Leakage [leloykun]
- Defines three TTT categories: (1) cheating (train+test on val), (2) token-stream TTT (problematic at scale), (3) document-independent TTT (valid — score-first per document)
- 9 PRs flagged as potentially leaking; only PR #152 closed
- Community note: AI agents monitoring others' PRs creates "trickledown" of flawed methods
- **Key rule**: Every token must be scored under inference_mode BEFORE any weight update

**Issue #43** — Tokenizer Size Counting [DouglasOrr]
- Custom tokenizer files count toward 16MB; default SP-1024 tokenizer is free
- Creates tension: large-vocab models get embedding overhead but tokenizer is "free"
- Resolution: custom vocabulary files count; only the official tokenizer is exempt

**Issue #202** — Share Weights Proposal [mihaibujanca]
- Weights not shared; forces re-training from scratch to evaluate others' work
- Supported by contributors including collaborators (0hq: "I agree, we can use GitHub since files are small")
- **Insight**: Competition generates a valuable diverse set of tiny LMs trained on identical data — ideal for interpretability and distillation research

**Issue #129** — Rules for Submission Flood [leloykun]
- One active record attempt per participant rule proposed
- Verification shadow companion (machine-readable evidence artifact) prototyped by EPLabsAI

**Issue #519** — Hardware Accessibility [yhy19]
- Fixed-step mode and fixed-compute mode benchmarks proposed as H100-agnostic alternatives
- MThePF training at exactly N steps (20,000) would separate algorithmic quality from hardware speed
- Implementation: torch.profiler FLOPs counting

---

### Notable Forks

**alientony/parameter-golf-Cross-Atten-RNN** — Cross-Attention + RNN hybrid; technical details not publicly documented beyond repo name; concept: combine attention for long-range with RNN for sequential efficiency.

**EV3KevinDEV/parameter-golf-qwen** — Qwen architecture adaptation; no detailed results documented in public README.

**gargraman/parameter-golf-slm** — SLM approach; fork of baseline with leaderboard showing 1.1748 BPB via spectral initialization + residual mixing (aligns with PR #60).

---

### Non-Record Submissions with Notable Insights

**PR #344 / PR #432** — Autoresearch approaches
- Automated 8-hour search on 1xH100 (PR #344): 28.7M params, 1.1383 BPB using score-first TTT
- Automated 2-campaign 188-run search on 1x5090 (PR #432): Found "allocation strategy > raw scaling"; losing approaches included blunt scaling, sequence-length expansion, broad precision relaxation
- **Insight**: Autoresearch pipelines from the NanoGPT speedrunning community (arjun-krishna1, jadechip, aamodbhatt) consistently contributed to the early leaderboard

**PR #496** — O(n) field LM, no attention
- 2L FP16 model without attention: val_bpb ≈ 21.42 — confirmation that attention is load-bearing at this scale
- Transferability: LOW for results; confirms attention > field models at language modeling

**PR #352** — Memory tokens
- 64 learnable "scratchpad" embedding vectors overwriting first K positions: −0.014 BPB
- Independently motivated from CoPE/memory-augmented transformer literature
- Transferability: HIGH — simple architectural addition with measurable gain

**PR #399** — Parameter Banking + Parallel Muon
- Consolidates weight matrices into tensor banks for batched Newton-Schulz (1 call vs 66)
- Manual DDP bypass for bank parameters improves communication overlap
- Transferability: MEDIUM — relevant for multi-GPU training with Muon-type optimizers

---

## Appendix: Technique Taxonomy

| Technique | Category | Primary Source | Multi-Source Validated | Notes |
|---|---|---|---|---|
| Sliding window evaluation (stride=64) | Evaluation | Records/SlidingWindowEval, PR #60 | YES (50+ reproductions) | −0.032 BPB; largest single eval gain |
| Int6 per-row quantization + zstd-22 | Quantization | PR #70 | YES (100+ uses) | Foundation of all competitive submissions |
| MLP 3x expansion | Architecture | PR #70 | YES (100+ uses) | Funded by int6 headroom |
| FP16 tied embeddings | Quantization | Records/FP16Embed | YES (widespread) | Prevents dual-path quant error |
| BigramHash embedding | Data/Architecture | PR #164 | YES (50+ uses) | 4096–12288 buckets; n-gram statistics |
| SmearGate | Architecture | Records/smeargate | YES (30+ uses) | ~512 params; blend prev token embedding |
| OrthoInit + muP | Initialization | PR #135 | YES (30+ uses) | Accelerates early convergence |
| Muon WD=0.04 | Optimizer | PR #198 | YES (40+ uses) | Controls artifact size + generalization |
| Stochastic Weight Averaging (SWA) | Ensemble | PR #198 | YES | Replaced by EMA in later records |
| EMA (decay=0.997) | Ensemble | PR #287 | YES (multiple) | Better than SWA by ~0.003 BPB |
| Tight SWA (final ~600 steps only) | Ensemble | PR #374 | YES (few) | Zero SWA penalty |
| XSA (Exclusive Self Attention) | Architecture | PR #287 | YES (10+ uses) | Zero params; −0.005 BPB; conflicts with TTT |
| U-Net skip connections | Architecture | Multiple records | YES (widespread) | Part of baseline architecture |
| Extended warmdown | Optimizer | Records/Warmdown | YES | Better quant distributions |
| FlashAttention 3 (Hopper) | Speed | PR #164 | YES | +15-20% steps on H100 |
| Partial RoPE (16/64 dims) | Architecture | PR #315 | YES (5+ uses) | Zero params; −0.002 BPB |
| LN Scale 1/sqrt(layer+1) | Initialization | PR #315 | YES (5+ uses) | Zero params; depth stabilization |
| QAT via STE | Quantization | Records/QAT | YES (30+ uses) | Eliminates quant roundtrip gap |
| Late QAT (final phase only) | Quantization | PR #374 | YES | Can be dead code with torch.compile |
| GPTQ-lite (5 percentiles) | Quantization | PR #379 | YES (3+ uses) | −0.0006 BPB; zero training cost |
| Full GPTQ (Hessian) | Quantization | PR #528 | YES (few) | −0.0024 BPB; 256-sample calibration |
| Mixed int5 MLP / int6 attn | Quantization | Records/Int5MLP | YES (10+ uses) | More capacity at same size |
| Gradient-guided adaptive quant | Quantization | PR #486 | SINGLE | Top 10% sensitivity→Int7 |
| Codebook + Huffman compression | Quantization | PR #532 | SINGLE | 21% better than int6+zstd |
| Arithmetic coding | Quantization | PR #538 | SINGLE | Approaches Shannon entropy |
| Weight entropy regularization | Optimizer/Ensemble | PR #459 | SINGLE | −0.028 BPB; better SWA smoothing |
| Value Residual (ResFormer) | Architecture | PR #487 | YES (5+ uses) | −0.015 BPB; 22 params |
| Gated Attention (per-head sigmoid) | Architecture | PR #487 | YES (5+ uses) | −0.003 BPB; 37K params |
| Catalytic Residuals | Architecture | PR #507 | SINGLE | Learned gates on residual contributions |
| Memory tokens | Architecture | PR #352 | SINGLE | −0.014 BPB; 64 learnable vectors |
| TrigramHash | Data/Architecture | PR #486 | SINGLE | Marginal gain over BigramHash |
| Shared Value Embedding | Architecture | PR #374 | YES (few) | Tie V projections across layers 9-10 |
| Legal score-first TTT | Test-time | PR #528, #473 | YES (multiple) | −0.010 to −0.020 BPB; AdamW > SGD |
| LoRA TTT (rank-8) | Test-time | Records/LoRA_TTT | YES (multiple) | Per-document isolation required |
| Parallel Muon + Parameter Banking | Speed | PR #399 | SINGLE | +3% throughput; batched Newton-Schulz |
| FP8 training (TransformerEngine) | Speed | PR #538 | SINGLE | +30-50% throughput |
| Depth recurrence | Architecture | Multiple PRs | YES | NEGATIVE with int6: 900x quant error |
| Depth recurrence | Architecture | Multiple PRs | YES | Needs int8+ to work; ~1.177 BPB ceiling |
| BitNet b1.58 ternary | Quantization | PR #367, #126 | YES | Standard techniques break in ternary regime |
| MoE routing | Architecture | PR #480 | YES (few) | NEGATIVE: routing overhead kills tiny models |
| INT4 quantization | Quantization | PR #375, #480 | YES | NEGATIVE: +0.065 BPB gap, too aggressive |
| MTP (Multi-Token Prediction) | Architecture | PR #375 | YES | NEGATIVE: +0.028 BPB due to step overhead |
| Cache LM (unigram) on web text | Test-time | PR #183 | SINGLE | NEGATIVE: FineWeb docs too short/diverse |
| XSA + TTT combined | Architecture | PR #303 | SINGLE | NEGATIVE: −0.016 BPB; redundant mechanisms |
| Adapt-first TTT | Test-time | Issue #402 | YES | NEGATIVE/INVALID: information leakage |
| Label smoothing | Regularization | PR #375 | SINGLE | NEGATIVE: no benefit at this scale |
| L1 regularization | Regularization | PR #375 | SINGLE | NEGATIVE: no benefit |
| SwiGLU (early attempts) | Architecture | Records/FP16Embed | SINGLE | NEGATIVE in early phase; positive in PR #507 |
| NorMuon (normalized Muon) | Optimizer | PR #438, multiple | YES | Alternative to Muon; comparable performance |
| Sequence length 2048 training | Data | Records/Seq2048 | YES | +0.008–0.018 BPB from better context |
| Sequence length 4096 training | Data | Records/Seq4096 | YES | Marginal gain over 2048; fewer steps |
| NTK-RoPE extrapolation | Architecture | Records/Warmdown | YES | Moderate extrapolation (1.375x) better than aggressive |
| torch.compile warmup | Speed | Issue #82 | YES | 3-4 min on H100; not counted against budget |
| Autoresearch pipelines | Methodology | PR #344, #432, #85 | YES | Effective for hyperparameter search; ~30 runs needed |

---

## Key Cross-Community Observations

**NanoGPT Speedrunning Cross-Pollination**: The Muon optimizer, BigramHash embedding, and sliding window evaluation all originated or were adapted from the NanoGPT speedrunning community. The record progression from 1.22 to 1.12 largely retraced techniques already developed there.

**Autoresearch Pattern**: Multiple participants (jadechip, arjun-krishna1, hydeh3r3, RyanLisse) ran automated hyperparameter search pipelines. The 1x5090 autoresearch found that "allocation strategy > raw component enrichment" — where you put compute matters more than what you use.

**torch.compile Dead Code**: Late QAT via STE in PR #315 was silently eliminated by torch.compile constant-folding at first trace. Any technique that depends on conditional behavior during the forward pass must verify it isn't optimized away. This is a general pitfall for any QAT implementation with torch.compile.

**Further Reading (External References)**:
- XSA paper: arXiv:2603.09078 (Exclusive Self Attention)
- MUD optimizer paper: arXiv:2603.17970 (Southworth & Thomas, triangular Gram preconditioning)
- ResFormer (Value Residual): independently cited by multiple contributors
- OpenAI Discord: #parameter-golf-discussions, #parameter-golf-announcements
- Pre-tokenized data: HuggingFace sproos/parameter-golf-tokenizers
