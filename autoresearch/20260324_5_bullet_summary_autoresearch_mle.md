# QKayV Roadmap: 5 Bullet Summary
*Synthesized from autoresearch techniques audit + MLE-Bench techniques audit*
*Date: 2026-03-24*

---

**1. Build tiered memory first — it's the highest-leverage unlock.**
Both research bodies converge on this: ML-Master 2.0's tiered memory (raw logs → phase summaries → cross-run insights) drove a 22.7% → 72.7% medal rate. The autoresearch ralph-loop showed the same pattern: agents without persistent memory re-discover the same findings and repeat the same failures across sessions. Build QKayV's memory layer in three tiers before anything else — L1 raw experiment outputs, L2 distilled phase summaries, L3 cross-run patterns that hold across experiments.

**2. Ship a two-agent architecture as your core product structure.**
Every high-performing system separates strategic planning from mechanical execution. LoongFlow's PES paradigm and R&D-Agent both show that planning-then-coding beats code-and-iterate — removing the planner increased time-to-solution by 52%. For QKayV: an **Analyzer agent** reads raw logs and produces structured summaries; a **Strategist agent** takes those summaries + knowledge retrieval and generates hypotheses. This is also what isolates the orchestrator from context window pollution.

**3. Add rich observability + early triage before adding more analysis features.**
PR #353 (autoresearch) showed 60% efficiency gain just from logging per-step training statistics instead of final metrics only — the agent learns *why* something failed, not just *that* it did. Paired with a 60-second structural triage checkpoint (spectral entropy of weight matrices), you eliminate ~80% of wasted compute on degenerate runs with zero new dependencies. These two things together will be your fastest trust-builder with early customers.

**4. Cross-experiment fusion is where QKayV's unique value lives.**
63% of actionable insights in MLE-Bench agents came from comparing *across* branches/experiments, not within a single run. This is the job that's currently impossible for humans managing dozens of parallel runs. Build a "failed approaches" memory the system actively queries before making recommendations, and surface patterns across experiments (e.g., "in 7 of your last 10 runs, LR > X caused instability in phase 2"). This is the core insight generation loop that justifies QKayV's existence.

**5. Gate your roadmap on hardware-specificity awareness — it will define your credibility.**
The single most important meta-finding across both research bodies: *strategies transfer across hardware, architectures do not.* Batch halving works universally (Z=0.517 certainty). SwiGLU wins on GB10, does nothing on H100. RoPE 200K collapsed on GH200. If QKayV gives the same recommendations to an H100 user and an RTX 4090 user, it will produce wrong answers and lose trust fast. Every recommendation the system generates must be tagged as universal vs. hardware-specific, and uncertainty estimates must accompany point predictions (don't say "increase LR" — say "based on 3 similar experiments on this hardware, 70% confidence this helps").
