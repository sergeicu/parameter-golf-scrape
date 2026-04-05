# QKayV Product Roadmap: Insights from Autoresearch Techniques Audit

*Prioritized recommendations for building QKayV's agentic ML training loop*

---

## Tier 1: Foundation Features (Build These First)

### 1. Persistent Memory with File-Based State

**Why first**: Context window degradation is the #1 killer of long sessions. The ralph-loop pattern (progress.md + next_ideas.md) solves this entirely — the agent starts fresh each iteration with clean context but retains knowledge.

**QKayV feature**: A `project_memory` layer that maintains:
- Compressed "what works / what doesn't" summary (~50 lines)
- Experiment history with keep/discard reasoning
- Ranked next-ideas queue

---

### 2. Early Structural Triage (60-second checkpoint)

**Why first**: Saves ~4 min/experiment on degenerate runs. 80% of wasted compute eliminated. Zero new dependencies.

**QKayV feature**: At the 1-minute mark, measure weight matrix effective rank (spectral entropy of SVD). If it drops >50% from initialization, kill the experiment immediately. This alone will make researchers trust the system.

---

### 3. Extended Training Statistics Logging

**Why first**: 60% efficiency gain (PR #353). The agent sees *why* something underperforms, not just *that* it does.

**QKayV feature**: Log per-step metrics, not just final val_bpb. Give the agent programmatic access to training dynamics (via Python analysis step).

---

## Tier 2: Core Differentiators

### 4. Multi-Objective Evaluation Guardrails

**Why first**: Goodhart's Law is real — agents gaming val_bpb is documented ( Discussion #322: agent replaced NN with alpha-beta search).

**QKayV feature**:
- Secondary downstream metrics (zero-shot accuracy on MMLU/Hellaswag)
- Code-enforced evaluation constraints (not prompt-only) — hide the eval harness from the agent
- Champion/challenger registry to prevent regressions

---

### 5. Adaptive Search Strategy State Machine

**Why first**: iii-hq/n-autoresearch showed 17 experiments/hour on dual RTX 4090 with this. The optimal search mode depends on current state.

**QKayV feature**: Auto-switch modes based on:
- `crash_rate > 50%` → exploit (don't explore while unstable)
- `plateau + near-misses` → combine (try pairs of near-wins)
- `plateau + no near-misses` → ablation (strip complexity)
- `keep_rate > 30%` → exploit (mine the current vein)
- Default → explore

---

### 6. Semantic Deduplication (Vector Memory)

**Why first**: Without it, multiple agents (or multiple sessions) independently test the same hypotheses.

**QKayV feature**: ChromaDB/FAISS store of prior experiment hypotheses + outcomes. Before proposing a new experiment, semantically search against history. Prevents wasted GPU time on re-discovered ideas.

---

## Tier 3: Advanced (After Foundations)

### 7. Multi-Agent Role Separation (Researcher/Skeptic/Synthesizer)

**Why**: Single-agent loop has no verification — can collude to satisfy prompts rather than verify logic.

**QKayV feature**:
- Researcher: proposes hypotheses, runs experiments
- Skeptic: tries to disprove findings before they're committed
- Synthesizer: commits validated findings to long-term memory

---

### 8. Cross-Session Knowledge Graph with Certainty Tracking

**Why**: Separates robust findings (Z > 0.5) from fragile ones. Hardware-specific findings (RoPE base, SwiGLU) must be tagged as such.

**QKayV feature**: Each finding gets a certainty Z-score updated via `η(Z) = sigmoid(10*(Z-0.5))`. Cross-platform findings are tagged — "batch halving works universally" vs "SwiGLU wins on GB10, neutral on H100."

---

## The Single Most Important Insight for QKayV

> **"Strategies transfer across hardware; architectures do not."** (Discussion #137, #195)

Your product must make this explicit. Researchers running on H100 vs GB200 vs consumer RTX will get *different optimal configs*. The loop must track which findings are universal vs hardware-specific, or researchers will chase dead ends.

---

## Bottom Line Priority Order

1. **Persistent memory + context isolation** (subagent for mechanical work)
2. **Early triage + rich observability** (extended stats)
3. **Multi-objective evaluation** (guard against metric gaming)
4. **Adaptive search state machine**
5. **Semantic deduplication** (vector memory)
6. **Multi-agent roles** (Researcher/Skeptic/Synthesizer)
7. **Cross-session certainty-tracked knowledge graph**

The first three give you 80% of the value with 20% of the complexity.
