# Parameter Golf Insights for QKayV Agentic Layer
*Generated from parameter_golf_research_report.md analysis*

---

## Core Insight: The Right Metric Is Everything

The report's most important meta-finding: **"At 86ms/step, every 1ms of per-step overhead costs ~0.006 BPB — techniques must be evaluated by throughput-adjusted delta."** Most ML tools evaluate quality in isolation. Your product should automatically compute **throughput-quality tradeoffs** rather than raw metrics.

---

## What Parameter Golf Reveals About the Research Loop Itself

### 1. The Stacking Order Was Discovered, Not Obvious

Progress wasn't linear. The community found:
- **Foundation first**: int6 + sliding window eval + MLP3x had to come before anything else (80% of competitive submissions built on this stack)
- **Layering rules**: You can't add XSA + TTT together (they conflict), but you CAN stack XSA + Partial RoPE + LN Scale (they're orthogonal)
- **Constraint propagation**: The 16MB budget constrained quantization choice, which constrained architecture choices, which constrained optimizer choices

**For an agentic layer**: This suggests a **dependency graph** of techniques where the agent must satisfy preconditions before suggesting additions. "What can I add given what I already have?"

### 2. Negative Results Were As Important As Positive Ones

The report documents ~15 systematic failure modes with root causes:
- MTP failed because step-time overhead > quality gain (not because the idea was wrong)
- QAT + EMA failed because of throughput interaction, not quality
- int5 MLP failed on well-trained models (only works undertrained)
- Cache LM failed on short diverse documents (only works on long homogeneous ones)

**For an agentic layer**: A research agent needs a **failure mode knowledge base** — not just "what works" but "what fails and why under which conditions." The report is essentially a taxonomy of failure modes.

### 3. The Measurement Process Itself Was Research

The single largest gain (−0.032 BPB) came from **changing how they measured**, not what they trained:
- Sliding window eval (stride=64 instead of full-context jump)
- GPTQ-lite (clipping at min-MSE instead of row-max)
- Throughput-adjusted BPB delta as the real metric

**For an agentic layer**: This suggests the agent must be able to **question the evaluation methodology itself** — not just optimize within the given measurement framework, but recognize when the framework is producing misleading signals.

### 4. The Search Strategy Evolved

The report shows a clear evolution in search methodology:
- **Phase 1** (early records): One variable at a time ablation
- **Phase 2** (mid records): Systematic stacking of validated components
- **Phase 3** (later records): Automated hyperparameter search (188 runs finding "allocation > enrichment")
- **Phase 4** (SOTA): Joint optimization over architecture + quantization + optimizer + TTT

**For an agentic layer**: The agent needs to know **which search strategy to apply when**. At early stages: ablate components individually. At mature stages: run joint optimization over the remaining degrees of freedom.

### 5. Information Leakage Was a Real Failure Mode

Issue #402 documented 9 PRs flagged for TTT violations — training on validation data. The community had to **define formal categories** of what constitutes leakage.

**For an agentic layer**: The agent must understand **research integrity constraints** — what separates valid test-time adaptation from data leakage. This isn't just technical; it's a methodological boundary the agent must enforce.

### 6. Single-Seed Results Were Treated as Suspicious

Multiple PRs noted "single-seed result, pending multi-seed validation." The community explicitly tracked reproducibility.

**For an agentic layer**: The agent should flag conclusions that lack statistical validation and design experiments that confirm or refute single-seed findings.

---

## What This Implies for the Agentic Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  RESEARCH AGENT CORE                                │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Technique   │  │ Failure Mode │  │ Search    │  │
│  │ Dependency  │  │ KB           │  │ Strategy  │  │
│  │ Graph       │  │ (known       │  │ Selector   │  │
│  │             │  │  failures +  │  │ (when to  │  │
│  │             │  │  conditions) │  │  ablate vs│  │
│  └─────────────┘  └──────────────┘  │  joint    │  │
│                                     │  search)  │  │
│  ┌─────────────┐  ┌──────────────┐  └───────────┘  │
│  │ Evaluation  │  │ Constraint   │                  │
│  │ Methodology │  │ Propagator   │                  │
│  │ Checker     │  │ (budget →    │                  │
│  │             │  │  quant →     │                  │
│  │             │  │  arch)        │                  │
│  └─────────────┘  └──────────────┘                  │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ Throughput-Adjusted Tradeoff Calculator     │    │
│  │ (the meta-skill the community learned)     │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### The Five Core Capabilities

1. **Dependency Resolution**: "Given technique X is in the stack, what can I validly add?"

2. **Failure Mode Lookup**: "Technique Y has failed in scenario Z because of mechanism M — does this apply here?"

3. **Search Strategy Router**: "Is this a component-ablation phase or a joint-optimization phase?"

4. **Evaluation Integrity Monitor**: "Is the measurement framework producing reliable signals?"

5. **Constraint Propagator**: "How does the 16MB budget constrain quantization choice, which constrains...?"

---

## The Key Insight: The Agentic Layer Is a Research Process Automator, Not a Technique Suggestion Engine

The difference:
- **Technique suggestion**: "Try EMA, it improves by 0.003 BPB"
- **Research process automation**: "Your current stack is in phase 2 (stacking validated components). The next action should be either (a) ablation of component X to confirm it's essential, or (b) running joint optimization over the remaining degrees of freedom. However, before doing either, verify your evaluation methodology isn't introducing measurement error — your sliding window stride appears suboptimal based on your context length."

The parameter golf community succeeded because they **automated the search over technique combinations** while maintaining **rigorous evaluation methodology**. The agentic layer for QKayV should focus on automating the *decision process* of research — what to try next, in what order, with what validation — not just the content of what to try.
