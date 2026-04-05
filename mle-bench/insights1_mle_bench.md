# Key Insights from MLE-Bench for Building QKayV

## Core Finding

The training loop is fundamentally a **search problem** — you're searching the space of data, architectures, hyperparameters, and training strategies. The MLE-Bench agents that improved from 17% → 64% didn't just get better at coding; they got better at **search + memory + reflection**. QKayV lives precisely in this "analyze logs → form hypotheses → decide what to try next" phase.

---

## Top 5 Agentic Features to Build First (Ranked by Leverage)

### 1. Tiered Memory with Progressive Distillation *(Highest Leverage)*

ML-Master 2.0 showed 22.7% → 72.7% by moving from raw traces to tiered summaries. For QKayV this maps directly:

- **L1 (Working)**: Raw experiment outputs, logs, metric curves
- **L2 (Distilled)**: Phase summaries — what changed, what improved, what degraded
- **L3 (Synthesized)**: Cross-run insights — patterns that hold across experiments

Raw logs to an LLM are overwhelming. The distillation chain is what makes analysis actually useful.

---

### 2. Post-Iteration Structured Reflection ("Why did this fail?")

LoongFlow's Summary module eliminated cyclical errors and improved sample efficiency by 60%. For QKayV:

- After each experiment run, auto-generate: what hypothesis was tested, what the outcome was, and **why** it diverged from expectation
- This prevents researchers from repeating the same mistake across dozens of runs
- The key architectural pattern: a dedicated "reviewer" role that analyzes outputs separately from the executor

---

### 3. Cross-Experiment Fusion

MLEvolve and MARS+ both showed that ~63% of actionable insights came from **comparing across branches/iterations** rather than within a single run. QKayV should:

- Surface patterns that appear across multiple experiments (e.g., "in 7 of your last 10 runs, increasing learning rate beyond X caused instability in phase 2")
- Build a "failed approaches" memory that the system actively retrieves against before recommending next steps

---

### 4. Knowledge Retrieval for Domain Context

MLE-STAR and KAPSO showed step-function improvements by pulling external knowledge (Kaggle solutions, papers, documentation). For QKayV:

- When analyzing logs, retrieve relevant literature/techniques for similar tasks/datasets
- "Your loss curve pattern matches papers [X, Y] which used technique Z — want me to explain?"
- This compensates for parametric knowledge gaps about specific techniques

---

### 5. Adaptive Exploration vs. Exploitation Switching

MLEvolve's time-aware switching (explore early, exploit late) is directly applicable to the research workflow:

- Early in a research direction: favor exploration (try diverse hypotheses)
- Late in a direction / when time-constrained: favor exploitation (refine best approach)
- QKayV should track "staleness" of research directions and recommend when to pivot vs. double down

---

## What NOT to Prioritize (Yet)

- **Multi-model ensembles** — Disarray's 77.8% required 4 frontier models and test-set feedback. Expensive and not applicable to your use case.
- **Full MCTS/evolutionary search** — that's for the execution phase (MLEvolve, MARS+). QKayV is for the analysis phase.
- **Git-native branching** — useful for reproducibility but not a core differentiator for insight generation.

---

## The Unresolved Problem (Watch For)

AIRA-dojo found a persistent **9-13% validation-to-test generalization gap** — agents overfit to cross-validation proxies. For QKayV, this means: any hypothesis the system generates should come with **uncertainty estimates**, not just point predictions. Don't just say "increase learning rate" — say "based on 3 similar experiments, this has a 70% confidence interval."

---

## Immediate Implementation Recommendation

Start with a **2-agent architecture**:

1. **Analyzer Agent**: Reads raw experiment outputs, produces structured summaries (L2 memory)
2. **Strategist Agent**: Takes L2/L3 memory + knowledge retrieval, generates hypotheses and recommendations

Connect them with the tiered memory system. Build the cross-experiment fusion and external retrieval on top. This mirrors the PES (Plan-Execute-Summary) paradigm that LoongFlow validated — the single highest-leverage architectural pattern in the report.
