# MLE-Bench Agent Techniques Audit: Comprehensive Report

**Date:** March 22, 2026
**Scope:** All agents on the MLE-Bench leaderboard, their repositories, and associated papers
**Baseline reference:** AIDE (o1-preview) — 17.1% "Any Medal" (Oct 2024)
**Current top performer:** Famou-Agent 2.0 — 64.4% (main leaderboard); Disarray — 77.8% (with test-set feedback)

---

# Level 1: Executive Key Insights

## Higher Performance

**1. Upgrading from greedy tree search to MCTS/evolutionary search yields +12-15% absolute improvement over baseline.**
Every agent scoring above 30% uses non-greedy search. ML-Master's MCTS with UCB selection, AIRA-dojo's systematic comparison of search policies, and MLEvolve's Monte Carlo Graph Search all demonstrate this. AIRA-dojo's ablation shows MCTS at 47% vs greedy at 39.8%. The principle: treat ML engineering as a search problem and invest in the search policy, not just the solution generator. [Source: ML-Master paper, AIRA-dojo paper arXiv:2507.02554]

**2. Separating strategic planning from tactical code execution is the single highest-leverage architectural change.**
R&D-Agent's Research/Development phase separation, LoongFlow's PES (Plan-Execute-Summary) paradigm, and ML-Master 2.0's hierarchical planning all show that agents that think before coding dramatically outperform agents that code-and-iterate. R&D-Agent's ablation shows removing dynamic planning causes the largest single-component performance drop. LoongFlow's planner removal increased time-to-solution by 52% (9.67h → 14.67h). [Source: R&D-Agent paper, LoongFlow paper arXiv:2512.24077]

**3. Structured memory with tiered abstraction enables 92.7% relative improvement (ML-Master 1.0 → 2.0: 29.3% → 56.4%).**
ML-Master 2.0's Hierarchical Cognitive Caching (HCC) uses three tiers: L1 (working memory of execution traces), L2 (distilled phase summaries), L3 (persistent cross-task wisdom). Ablation shows L1-only achieves 22.7% medal rate vs full HCC at 72.7% on MLE-Bench-Lite. The key insight: raw context overwhelms the model; progressive distillation preserves signal while managing context budgets. [Source: ML-Master 2.0 paper arXiv:2601.10402]

**4. Cross-branch fusion rescues stalled search by merging insights from parallel solution branches.**
MLEvolve's cross-branch fusion identifies top-performing nodes across different branches and merges their insights when all branches stagnate. CAIR MARS+ reports that 63% of utilized lessons originated from cross-branch transfer. This prevents the common failure mode of converging to a single approach and getting stuck. [Source: MLEvolve repo, MARS paper arXiv:2602.02660]

**5. External knowledge retrieval (Kaggle solutions, papers, documentation) provides a step-function improvement over relying on parametric knowledge alone.**
CAIR MLE-STAR was the first to systematically pull from external sources. Leeroo KAPSO extends this with a knowledge graph and "Leeroopedia MCP" domain knowledge wiki. Agents with retrieval consistently outperform agents without, because ML engineering requires domain-specific knowledge (library APIs, competition tricks, data-specific preprocessing) that LLMs don't reliably encode. [Source: MLE-STAR leaderboard entry, KAPSO paper arXiv:2601.21526]

**6. Multi-model ensembles with diverse LLM backbones push the frontier another +13% absolute beyond single-model agents.**
Disarray ensembles Claude-Opus-4.5, Claude-Sonnet-4.5, GPT-5.2-Codex, and Gemini-3-Pro. The gain comes from model diversity: different LLMs have different strengths across competition types (vision, NLP, tabular). However, this technique has diminishing returns and high cost. [Source: Disarray leaderboard entry]

**7. Explicit post-iteration reflection ("why did this fail?") eliminates cyclical errors and improves sample efficiency by 60%.**
LoongFlow's Summary module performs multi-dimensional review after each iteration. Without it, agents repeat the same mistakes — one trial ran 35 hours and failed to break 0.95 threshold. With it, LoongFlow achieved equivalent results to OpenEvolve with 60% fewer LLM calls (258 vs 783 evaluations). The principle: reflection is not optional overhead; it's the mechanism that makes each iteration count. [Source: LoongFlow paper arXiv:2512.24077]

## Better Generalization

**8. Component-wise exploration (optimizing data/model/training/postprocessing independently) reduces search space and improves attributability.**
CAIR MLE-STAR and MARS+ decompose the ML pipeline into independently optimizable components rather than modifying everything at once. This mirrors good ML engineering practice and makes it easier to identify which changes helped. The "Design-Decompose-Implement" pipeline in MARS+ achieved 62.67% overall. [Source: MARS+ paper arXiv:2602.02660]

**9. Cold-start model recommendations per task category eliminate wasted early iterations.**
MLEvolve provides pre-configured model recommendations by task type (image classification, tabular, NLP). This avoids the common failure mode of spending half the time budget trying obviously inappropriate approaches (e.g., tabular models for image tasks). The principle: encode domain knowledge as priors, don't force the agent to rediscover it. [Source: MLEvolve repo]

**10. Adaptive code generation mode selection (full rewrite vs. diff patching vs. multi-agent pipeline) improves robustness across task complexity.**
MLEvolve switches between single-pass generation (for exploration), stepwise multi-agent pipeline (for complex tasks), and SEARCH/REPLACE diff patching (for refinement). LoongFlow's "Fuse Mode" switches between single-turn chat (fast) and multi-turn ReAct (reliable) based on task complexity. One-size-fits-all code generation fails because tasks vary in complexity. [Source: MLEvolve repo, LoongFlow paper]

**11. Maintaining behavioral diversity via MAP-Elites prevents premature convergence and improves performance on diverse task types.**
LoongFlow uses MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) to maintain diverse solutions across behavioral dimensions. Entropy-regularized Boltzmann selection automatically adjusts exploration temperature. Without diversity mechanisms, evolutionary agents converge to one solution type and fail on tasks requiring different approaches. [Source: LoongFlow paper arXiv:2512.24077]

**12. The 9-13% validation-to-test generalization gap is a persistent unsolved problem across all agents.**
AIRA-dojo discovered that agents consistently overfit to cross-validation proxy metrics. Oracle analysis shows +16.6% potential improvement if agents could perfectly select their best submission. No current framework has closed this gap. The implication: any production ML automation system needs robust held-out evaluation or uncertainty quantification. [Source: AIRA-dojo paper arXiv:2507.02554]

**13. Git-native experimentation with branch-per-attempt provides reproducibility and provenance tracking.**
Leeroo KAPSO isolates each attempt as a git branch, producing reproducible artifacts and preserving provenance across iterations. This pattern enables post-hoc analysis of what changed between successful and failed experiments. [Source: KAPSO repo]

## More Efficient

**14. Execution infrastructure quality alone yields +10.7% absolute improvement — often more than agent logic changes.**
AIRA-dojo demonstrated that improving error handling, metric extraction, and sandboxing — without changing agent logic — boosted AIDE's score from 35.2% to 45.9% (a 30% relative improvement). The principle: before optimizing the agent, optimize the environment it operates in. Better error messages and reliable metric extraction make every agent iteration more productive. [Source: AIRA-dojo paper arXiv:2507.02554]

**15. Cost-aware search planning (balancing performance against compute cost) improves efficiency without sacrificing outcomes.**
CAIR MARS+ implements cost-constrained MCTS that explicitly accounts for computational expense when deciding which branches to expand. MLEvolve achieves #1 in the 12-hour budget category (matching 24-hour agents at 61.3%), demonstrating that smarter search beats longer search. [Source: MARS paper, MLEvolve leaderboard entry]

**16. Time-aware explore-exploit switching automatically allocates early time to exploration and late time to exploitation.**
MLEvolve's piecewise exploration decay and automatic stagnation detection dynamically shift strategy based on remaining time budget. This prevents agents from wasting late-stage time on risky exploration when they should be polishing their best solution. [Source: MLEvolve repo]

**17. Hierarchical context management reduces peak context from >200k tokens to ~70k while preserving decision-relevant information.**
ML-Master 2.0's HCC progressively compresses execution traces into summaries and then into reusable wisdom. This directly addresses the context window bottleneck that limits all LLM-based agents. The principle: don't feed raw logs to the LLM — distill them into actionable summaries. [Source: ML-Master 2.0 paper arXiv:2601.10402]

**18. Using cheaper models for code execution and expensive reasoning models only for planning dramatically reduces cost.**
R&D-Agent pairs o3 (expensive, for research/planning) with GPT-4.1 (cheaper, for implementation/debugging). The multi-trace configuration achieved 24% overall, notably excelling at high-complexity tasks (20% vs 10% baseline). This model-mixing pattern is applicable to any multi-step agent system. [Source: R&D-Agent paper arXiv:2505.14738]

---

# Level 2: Technical Report

## Leaderboard Agents (Ranked by Performance)

### Main Leaderboard (No Test-Set Feedback)

---

### Famou-Agent 2.0 — 64.4% "Any Medal"
- **Repository/Paper Link**: Not publicly available (closed source)
- **LLM Backbone**: Gemini-3-Pro-Preview
- **Performance**: Low 80.3% | Med 64.0% | High 42.2% | All 64.4%
- **Runtime**: 24 hours
- **Date**: 2026-02
- **Technique(s) identified**:
  - Multi-agent architecture: Details unavailable (closed source)
  - Search/exploration: Believed to use evolutionary or MCTS-based approach given performance level
  - Memory: Experience-driven memory with FAISS/BM25 retrieval (per Gemini analysis)
  - Code generation: Unknown specifics
- **Performance delta vs AIDE**: +47.3% absolute (17.1% → 64.4%)
- **Key improvements vs baseline**: Upgraded from v1.0 (43.6%) to v2.0 with +20.8% absolute gain, suggesting major architectural changes between versions
- **Estimated transferability**: Unknown (closed source limits assessment)
- **Source**: MLE-Bench leaderboard

---

### AIBuildAI — 63.1% "Any Medal"
- **Repository/Paper Link**: [github.com/aibuildai/AI-Build-AI](https://github.com/aibuildai/AI-Build-AI)
- **LLM Backbone**: Claude-Opus-4.6
- **Performance**: Low 77.3% | Med 61.4% | High 46.7% | All 63.1%
- **Runtime**: 24 hours
- **Date**: 2026-02
- **Technique(s) identified**:
  - Architecture: Iterative agent loop — problem analysis → model design → code implementation → training → hyperparameter tuning → evaluation → improvement
  - Search/exploration: Recursive self-improvement loops (per Gemini analysis)
  - Code generation: Full pipeline generation with iterative refinement
  - Notable: Highest High-difficulty score (46.7%) among main leaderboard agents
- **Performance delta vs AIDE**: +46.0% absolute
- **Key improvements vs baseline**: Strong performance on high-difficulty tasks suggests robust problem decomposition
- **Estimated transferability**: Medium — iterative loop pattern is universal but specific implementation details are limited
- **Source**: GitHub repo, leaderboard

---

### CAIR MARS+ — 62.7% "Any Medal"
- **Repository/Paper Link**: [arXiv:2602.02660](https://arxiv.org/abs/2602.02660) (MARS framework paper)
- **LLM Backbone**: Gemini-3-Pro-Preview
- **Performance**: Low 78.8% | Med 60.5% | High 44.4% | All 62.7%
- **Runtime**: 24 hours
- **Date**: 2026-02
- **Technique(s) identified**:
  - Multi-agent architecture: Modular agent with specialized roles
  - Search/exploration: Cost-constrained MCTS — explicitly balances performance against computational expense
  - Memory: **Comparative Reflective Memory** — analyzes solution differences across branches to extract high-signal insights; 63% of utilized lessons came from cross-branch transfer
  - Code generation: "Design-Decompose-Implement" pipeline for managing complex codebases
  - Evaluation: Budget-aware planning with explicit cost/performance tradeoff
- **Performance delta vs AIDE**: +45.6% absolute
- **Key improvements vs baseline**: Budget-aware search + cross-branch reflective memory + modular decomposition
- **Estimated transferability**: High — cost-aware search and reflective memory are broadly applicable patterns
- **Source**: MARS paper arXiv:2602.02660, leaderboard

---

### MLEvolve — 61.3% "Any Medal"
- **Repository/Paper Link**: [github.com/InternScience/MLEvolve](https://github.com/InternScience/MLEvolve)
- **LLM Backbone**: Gemini-3-Pro-Preview
- **Performance**: Low 80.3% | Med 57.9% | High 42.2% | All 61.3%
- **Runtime**: 12 hours (half the typical 24h budget — best efficiency)
- **Date**: 2026-02
- **Technique(s) identified**:
  - Multi-agent architecture: Three-agent system — Code Agent (generation/refinement), Feedback Agent (evaluation/suggestions), Planning Agent (exploration strategy coordination)
  - Prompting strategy: Adaptive dispatch across three planning paradigms
  - Search/exploration: **Monte Carlo Graph Search (MCGS)** with:
    - Piecewise exploration decay (shifting from exploration to exploitation over time)
    - Time-aware explore-exploit switching based on remaining budget
    - Automatic stagnation detection triggering recovery mechanisms
    - **Cross-branch fusion** — merges insights from top nodes across branches when progress stalls
    - Trajectory-aware evolution leveraging full branch improvement history
  - Memory: **Experience-driven global memory** recording plan/code/metrics/success-failure for every node. Hybrid BM25 + FAISS retrieval. Different agents query memory differently to encourage novel approaches.
  - Code generation: Three adaptive modes:
    1. Single-pass direct generation (for exploration)
    2. Stepwise multi-agent pipeline (for complex tasks)
    3. Incremental SEARCH/REPLACE diff patching (for refinement)
  - Cold-start: Pre-configured model recommendations per task category
- **Performance delta vs AIDE**: +44.2% absolute
- **Key improvements vs baseline**: MCGS search + experience memory + cross-branch fusion + adaptive code gen + cold-start priors. Achieves comparable scores to 24h agents in only 12h.
- **Estimated transferability**: High — MCGS, experience memory, and adaptive code generation are all broadly applicable
- **Source**: GitHub repo

---

### PiEvolve (24h) — 61.3% "Any Medal"
- **Repository/Paper Link**: Not publicly available (Fractal, closed source)
- **LLM Backbone**: Gemini-3-Pro-Preview
- **Performance**: Low 80.3% | Med 58.8% | High 40.0% | All 61.3%
- **Runtime**: 24 hours (also has 12h variant at 52.0%)
- **Date**: 2026-01
- **Technique(s) identified**:
  - Architecture: Evolutionary agentic engine
  - Search/exploration: Continuous optimization — iteratively evolves candidate solutions until compute budget exhausted
  - Memory: Priority-based sampling with decay to avoid local optima
  - First agent to surpass 60% overall and 80% on MLE-Bench-Lite
- **Performance delta vs AIDE**: +44.2% absolute
- **Key improvements vs baseline**: Evolutionary approach with intelligent memory sampling
- **Estimated transferability**: Medium — evolutionary approach is general but specific details are proprietary
- **Source**: Fractal press release, leaderboard

---

### Famou-Agent 1.0 — 43.6% "Any Medal"
- **Repository/Paper Link**: Not publicly available
- **LLM Backbone**: Gemini-2.5-Pro
- **Performance**: Low 62.1% | Med 36.8% | High 33.3% | All 43.6%
- **Runtime**: 24 hours
- **Date**: 2025-10
- **Key insight**: v1.0 → v2.0 jump (+20.8% absolute) is one of the largest single-agent improvements, suggesting major architectural revision between versions
- **Source**: Leaderboard

---

### ML-Master 2.0 — 56.4% "Any Medal"
- **Repository/Paper Link**: [github.com/sjtu-sai-agents/ML-Master](https://github.com/sjtu-sai-agents/ML-Master), [arXiv:2601.10402](https://arxiv.org/abs/2601.10402)
- **LLM Backbone**: Deepseek-V3.2-Speciale
- **Performance**: Low 75.8% | Med 50.9% | High 42.2% | All 56.4%
- **Runtime**: 24 hours
- **Date**: 2025-12
- **Technique(s) identified**:
  - Architecture: Hierarchical planning with parallel exploration directions
  - Search/exploration: Structured plans with m exploration directions × q concrete suggestions, executed in parallel
  - Memory: **Hierarchical Cognitive Caching (HCC)** — three-tier system:
    - **L1 Cache (Evolving Experience)**: Working memory of execution traces. Discarded upon phase completion to prevent context saturation.
    - **L2 Cache (Refined Knowledge)**: LLM-distilled phase summaries preserving key judgments and decision rationales.
    - **L3 Cache (Prior Wisdom)**: Persistent cross-task memory with embedding-paired task descriptors for similarity-based retrieval.
  - Context migration: Three protocols:
    - Context Prefetching: Retrieves similar prior wisdom via cosine similarity thresholding
    - Context Hit: Cache-like retrieval (L1 if available, fallback to L2)
    - Context Promotion: Two-stage LLM-based abstraction compressing traces → summaries → reusable wisdom
  - Code generation: Parallel exploration with consolidation at phase boundaries
  - Evaluation: Tracks validity, above-median, and medal rates
- **Ablation results (MLE-Bench-Lite)**:
  - L1 only: 22.7% medal rate
  - L2 (raw context): 59.1%
  - L3 (no transfer): 54.5%
  - **Full HCC: 72.7%** — each tier contributes synergistically
- **Performance delta vs AIDE**: +39.3% absolute (and +92.7% relative improvement over ML-Master 1.0)
- **Key improvements vs baseline**: Three-tier memory hierarchy enabling progressive knowledge distillation. Reduced peak context from >200k to ~70k tokens while maintaining performance.
- **Estimated transferability**: High — tiered memory abstraction pattern applies to any long-horizon LLM agent task
- **Source**: Paper arXiv:2601.10402, GitHub repo

---

### Leeroo (KAPSO) — 50.7% "Any Medal"
- **Repository/Paper Link**: [github.com/Leeroo-AI/kapso](https://github.com/Leeroo-AI/kapso), [arXiv:2601.21526](https://arxiv.org/abs/2601.21526)
- **LLM Backbone**: Gemini-3-Pro-Preview
- **Performance**: Low 68.2% | Med 44.7% | High 40.0% | All 50.7%
- **Runtime**: 24 hours
- **Date**: 2025-12
- **Technique(s) identified**:
  - Architecture: Four-pillar system — Evolve (tree search + coding agents), Learn (knowledge graph ingestion), Research (web discovery), Deploy (solution deployment)
  - Search/exploration: Tree search combined with knowledge-graph-informed coding agents
  - Memory: **Knowledge graph** ingesting repositories and research papers + **workflow-aware cognitive memory** storing experiment trace lessons
  - External knowledge: **Leeroopedia MCP** — ML/Data domain knowledge wiki supplying expert-level guidance during ideation
  - Code generation: Git-native experimentation — each attempt isolated as a branch for reproducibility
  - Research integration: Cascaded retrieval (WSR with PFR fallback + ERA augmentation) from heterogeneous sources
- **Performance delta vs AIDE**: +33.6% absolute
- **Key improvements vs baseline**: Knowledge-grounded synthesis + domain knowledge wiki + git-native experimentation
- **Estimated transferability**: High — knowledge graph + domain knowledge retrieval pattern is broadly applicable; git-native experimentation is best practice
- **Source**: GitHub repo, paper arXiv:2601.21526

---

### Thesis — 48.4% "Any Medal"
- **Repository/Paper Link**: [thesislabs.ai](https://thesislabs.ai/writings/sota-mle-bench) (closed source)
- **LLM Backbone**: GPT-5-Codex
- **Performance**: Low 65.2% | Med 45.6% | High 31.1% | All 48.4%
- **Runtime**: 24 hours
- **Date**: 2025-11
- **Technique(s) identified**:
  - Architecture: End-to-end ML engineering loop (problem understanding → data processing → training → iteration)
  - Key: First major demonstration of GPT-5-Codex on ML engineering tasks
  - Limited architectural details (proprietary)
- **Performance delta vs AIDE**: +31.3% absolute
- **Estimated transferability**: Unknown (closed source)
- **Source**: Thesis Labs blog, leaderboard

---

### CAIR MLE-STAR-Pro — 44.0% / 38.7% "Any Medal"
- **Repository/Paper Link**: Not publicly available (Google Cloud AI Research)
- **LLM Backbone**: Gemini-2.5-Pro
- **Performance (v1.5)**: Low 68.2% | Med 34.2% | High 33.3% | All 44.0%
- **Runtime**: 24 hours (v1.5) / 12 hours (v1.0)
- **Date**: 2025-11
- **Technique(s) identified**:
  - External knowledge retrieval: First to systematically pull from Kaggle solution notebooks, papers, and documentation
  - Component-wise exploration: Optimizes data preprocessing, model architecture, training, and postprocessing independently
  - Gemini backbone: Demonstrated strong Gemini performance on ML engineering tasks
- **Performance delta vs AIDE**: +26.9% absolute
- **Estimated transferability**: High — component-wise optimization and external retrieval are universally applicable
- **Source**: Leaderboard

---

### Operand — 39.6% "Any Medal"
- **Repository/Paper Link**: Not publicly available
- **LLM Backbone**: GPT-5
- **Performance**: Low 63.6% | Med 33.3% | High 20.0% | All 39.6%
- **Runtime**: 24 hours
- **Date**: 2025-10
- **Source**: Leaderboard only — no architectural details available

---

### InternAgent — 36.4% "Any Medal"
- **Repository/Paper Link**: Not publicly available
- **LLM Backbone**: DeepSeek-R1
- **Performance**: Low 62.1% | Med 26.3% | High 24.4% | All 36.4%
- **Runtime**: 12 hours
- **Date**: 2025-09
- **Source**: Leaderboard only — limited details. Predecessor to MLEvolve (InternScience lineage).

---

### R&D-Agent — 35.1% (GPT-5) / 30.2% (o3) / 22.4% (o1) "Any Medal"
- **Repository/Paper Link**: [github.com/microsoft/RD-Agent](https://github.com/microsoft/RD-Agent), [arXiv:2505.14738](https://arxiv.org/abs/2505.14738)
- **LLM Backbone**: GPT-5 / o3 + GPT-4.1 / o1-preview (multiple configurations)
- **Performance (GPT-5)**: Low 68.2% | Med 21.1% | High 22.2% | All 35.1%
- **Runtime**: 12-24 hours
- **Date**: 2025-05 to 2025-09
- **Technique(s) identified**:
  - Multi-agent architecture: **Dual-phase Research/Development separation**
    - Research phase (expensive reasoning model, e.g. o3): generates hypotheses, plans experiments, strategic reasoning
    - Development phase (cheaper model, e.g. GPT-4.1): implements, debugs, iterates on code
  - Prompting strategy: Dynamic planning — strategy adapts based on accumulated results, pivoting when lines of investigation fail
  - Memory: Structured memory of previous R&D loops (plan → implementation → result)
  - Code generation: Iterative debugging on sampled data, then full dataset evaluation
  - Multi-trace: Multiple parallel traces improve High-difficulty performance (20% vs 10% baseline)
- **Key ablation**: Removing dynamic planning caused largest single-component drop
- **Performance delta vs AIDE**: +18.0% absolute (GPT-5 config)
- **Estimated transferability**: High — R&D phase separation and model-mixing pattern are broadly applicable
- **Source**: GitHub repo, paper arXiv:2505.14738

---

### Neo Multi-Agent — 34.2% "Any Medal"
- **Repository/Paper Link**: Not publicly available
- **LLM Backbone**: Undisclosed
- **Performance**: Low 48.5% | Med 29.8% | High 24.4% | All 34.2%
- **Runtime**: 36 hours (longest runtime on leaderboard)
- **Date**: 2025-07
- **Key observation**: Despite the longest runtime (36h), underperforms agents with 12-24h budgets, demonstrating that more compute without better search strategy has diminishing returns
- **Source**: Leaderboard

---

### AIRA-dojo — 31.6% "Any Medal"
- **Repository/Paper Link**: [github.com/facebookresearch/aira-dojo](https://github.com/facebookresearch/aira-dojo), [arXiv:2507.02554](https://arxiv.org/abs/2507.02554)
- **LLM Backbone**: o3
- **Performance**: Low 55.0% | Med 22.0% | High 21.7% | All 31.6%
- **Runtime**: 24 hours
- **Date**: 2025-05
- **Technique(s) identified**:
  - Architecture: Modular three-layer framework — Tasks (execution environments), Solvers (agents = operators + search policies), Runners (parallelization up to 1000 agents)
  - Search/exploration: **Four systematically compared policies**: AIDE_GREEDY (39.8%), AIRA_GREEDY (45.5%), AIRA_MCTS (47.0%), AIRA_EVO (~45%). First rigorous policy comparison.
  - **Operator set (AIRA operators)**: Draft, Debug, Improve, Memory, Crossover. Complexity-adaptive code generation adjusting guidance (minimal/moderate/advanced) to task difficulty.
  - Think tokens: 2× increase in completion tokens via explicit thinking prompts
  - Memory operator: Tested but showed negligible effect in their ablation — nearly identical performance with/without (contradicts other findings)
  - Infrastructure: Apptainer containers, Slurm scheduler, W&B tracking, Hydra config
  - **Key finding**: Infrastructure improvements alone yielded +10.7% absolute (35.2% → 45.9%)
  - **Key finding**: 9-13% persistent generalization gap between validation and test performance
  - **Key finding**: Extended compute to 90h yielded only +6% more (peaked at 53%)
- **Performance delta vs AIDE**: +14.5% absolute
- **Estimated transferability**: Very High — this is primarily a research contribution with rigorous ablations. The decomposition into search policies + operators is a universal framework.
- **Source**: Paper arXiv:2507.02554, GitHub repo

---

### ML-Master 1.0 — 29.3% "Any Medal"
- **Repository/Paper Link**: [arXiv:2506.16499](https://arxiv.org/abs/2506.16499)
- **LLM Backbone**: DeepSeek-R1
- **Performance**: Low 48.5% | Med 20.2% | High 24.4% | All 29.3%
- **Runtime**: 12 hours
- **Date**: 2025-06
- **Technique(s) identified**:
  - Upgraded AIDE's greedy search to MCTS with UCB selection
  - Exploration-vs-exploitation balancing via UCB
  - Paired with reasoning model (DeepSeek-R1)
- **Performance delta vs AIDE**: +12.2% absolute
- **Key insight**: ~Equal contribution from MCTS upgrade and reasoning model backbone
- **Estimated transferability**: High — MCTS is a well-understood algorithm applicable to any search problem
- **Source**: Paper arXiv:2506.16499

---

### R&D-Agent (o1-preview) — 22.4% "Any Medal"
- See R&D-Agent section above. Earliest R&D-Agent configuration.

---

## Separate Category: Test-Set Feedback Advantage

### Disarray — 77.8% "Any Medal"
- **Repository/Paper Link**: Not available (Disarray.ai, closed source)
- **LLM Backbone**: Ensemble — Claude-Opus-4.5, Claude-Sonnet-4.5, GPT-5.2-Codex, Gemini-3-Pro-Preview
- **Performance**: Low 90.9% | Med 72.8% | High 71.1% | All 77.8%
- **Runtime**: 24 hours
- **Date**: 2026-02
- **Technique(s) identified**:
  - Multi-model ensemble: Four frontier LLMs providing model diversity
  - Known test label leakage exploitation on some benchmark tasks
  - "Context-First Integration" / "Agentic Context Engineering (ACE)" — manages context window budget by offloading heavy logs to external memory, feeding only delta updates to reasoning core (per Gemini analysis)
- **Performance delta vs AIDE**: +60.7% absolute
- **Key caveat**: Uses known test-set feedback advantage. Scores are not directly comparable to main leaderboard.
- **Estimated transferability**: Medium — ensemble technique is general but expensive; leakage exploitation is benchmark-specific
- **Source**: Leaderboard

---

### LoongFlow — 62.7% "Any Medal"
- **Repository/Paper Link**: [github.com/baidu-baige/LoongFlow](https://github.com/baidu-baige/LoongFlow), [arXiv:2512.24077](https://arxiv.org/abs/2512.24077)
- **LLM Backbone**: Gemini-3-Flash-Preview
- **Performance**: Low 77.3% | Med 63.2% | High 40.0% | All 62.7%
- **Runtime**: 24 hours
- **Date**: 2026-02
- **Technique(s) identified**:
  - Multi-agent architecture: **PES (Plan-Execute-Summary) cognitive paradigm** with three specialized roles:
    - **Planner**: Reviews full history, analyzes task, generates strategic blueprint
    - **Executor**: Implements plan with adaptive "Fuse Mode" (single-turn chat for fast/simple, multi-turn ReAct for complex)
    - **Summarizer**: Multi-dimensional review analyzing what worked, what didn't, and why
  - Search/exploration: **Hybrid evolutionary with cognitive structure**:
    - **Evolution Tree**: Global decision history enabling causal chain tracing
    - **MAP-Elites**: Maintains diverse solutions across behavioral dimensions, preventing premature convergence
    - **Entropy-regularized Boltzmann selection**: Auto-adjusting temperature — rises when diversity drops, falls when promising solutions emerge
  - Memory: **Multi-Structure Fusion Memory** — active reasoning context generation + structured insight persistence + cross-iteration knowledge accumulation
  - Code generation: Skill-driven methodology with custom skill packages encoding domain knowledge
  - Evaluation: Interactive web visualization showing code diffs and evolution tracking
- **Ablation results**:
  - Removing Planner: time-to-solution +52% (9.67h → 14.67h), stagnation below 0.96
  - Removing Summarizer: cyclical errors, one trial 35h without breaking 0.95
  - Fuse Mode (Chat + ReAct): highest asymptotic score (0.998) with optimal sample efficiency
  - Sample efficiency: 258 evaluations vs OpenEvolve's 783 (67% reduction)
- **Performance delta vs AIDE**: +45.6% absolute
- **Key improvements vs baseline**: PES cognitive loop + MAP-Elites diversity + entropy-regulated selection + 60% better sample efficiency
- **Estimated transferability**: Very High — PES paradigm is a universal cognitive architecture for any iterative optimization agent
- **Source**: Paper arXiv:2512.24077, GitHub repo

---

## Baseline Comparison

### AIDE (Original Baseline) — 17.1% "Any Medal"
- **Repository/Paper Link**: [github.com/wecoai/aideml](https://github.com/wecoai/aideml), [arXiv:2502.13138](https://arxiv.org/abs/2502.13138)
- **LLM Backbone**: o1-preview (best config); also tested with GPT-4o (8.6%), Claude-3.5-Sonnet (7.6%), Llama-3.1-405B (3.3%)
- **Performance (o1-preview)**: Low 35.9% | Med 8.5% | High 11.7% | All 17.1%
- **Runtime**: 24 hours
- **Technique(s)**:
  - Single-agent tree search in code space — Python scripts as nodes, LLM-generated patches create children
  - Three coding operators: drafting, debugging, improving
  - Metric-driven pruning — evaluates against metric, expands best-performing node
  - 20 improvement steps, 5 parallel candidates per step
  - Natural language task specification
- **Key strengths**: First to formalize agentic ML as tree search; 4× better than linear agents (OpenHands)
- **Failure modes that successors addressed**:
  1. **No memory across iterations** — each node generated from scratch, no learning from past failures
  2. **Greedy search policy** — always expands best-performing node, gets stuck in local optima
  3. **Single-agent** — one LLM does planning, coding, and debugging with no role separation
  4. **No external knowledge retrieval** — relies entirely on parametric knowledge
  5. **No reflection** — no analysis of why iterations fail, leading to repeated mistakes
  6. **No diversity maintenance** — search converges to single solution type
  7. **Raw context overload** — no context management, wastes tokens on irrelevant execution traces

---

## Cross-Agent Analysis

### Architecture Patterns

| Pattern | Agents Using It | Best Score | Verdict |
|---------|----------------|-----------|---------|
| Single-agent tree search | AIDE | 17.1% | Baseline, insufficient alone |
| MCTS/UCB search | ML-Master, AIRA-dojo, MARS+ | 62.7% | Consistent +12-15% over greedy |
| Evolutionary search | PiEvolve, LoongFlow, MLEvolve | 64.4% | Competitive with MCTS |
| Multi-agent role separation | MLEvolve, LoongFlow, R&D-Agent | 62.7% | Helps, especially with Planner/Executor split |
| Multi-model ensemble | Disarray | 77.8% | Expensive but effective (test-set advantage caveat) |
| Dual-phase R&D | R&D-Agent | 35.1% | Planning separation validated by ablation |

**Key finding**: Multi-agent systems that separate *roles* (planner vs executor vs reviewer) outperform those that just add more agents doing the same thing. Neo's 36-hour multi-agent system (34.2%) underperforms well-designed single-agent systems with good search.

### Prompting Evolution

| Era | Prompting Approach | Example Agent | Score Range |
|-----|--------------------|--------------|-------------|
| Baseline | Simple task description + code generation | AIDE | 8-17% |
| +Reasoning model | Chain-of-thought via reasoning model backbone | ML-Master 1.0 (DeepSeek-R1) | 29% |
| +Think tokens | Explicit thinking prompts doubling completion tokens | AIRA-dojo | 31.6% |
| +Structured planning | Strategic blueprint before coding | LoongFlow (PES), R&D-Agent | 35-63% |
| +Adaptive complexity | Adjusting prompt complexity to task difficulty | AIRA-dojo operators, MLEvolve | 47-64% |
| +Post-iteration reflection | Explicit "why did this fail?" analysis | LoongFlow Summary, MARS+ reflective memory | 56-63% |

**Key finding**: The evolution is from "generate code" → "reason then code" → "plan, code, reflect" → "plan, code, reflect, remember". Each layer compounds.

### Tool Usage

| Tool/Integration | Agents Using It | Impact |
|-----------------|----------------|--------|
| Python REPL / code execution | All agents | Table stakes |
| Containerized sandboxing (Apptainer/Docker) | AIRA-dojo, KAPSO | +10.7% from infra alone |
| W&B experiment tracking | AIRA-dojo | Better metric extraction |
| External knowledge retrieval | MLE-STAR, KAPSO (Leeroopedia) | Step-function improvement on domain tasks |
| Git-native branching | KAPSO | Reproducibility + provenance |
| Hybrid search (BM25 + FAISS) | MLEvolve | Better memory retrieval |
| Knowledge graph | KAPSO | Structured domain knowledge |
| Web search / research | KAPSO (Research pillar) | External information access |

**Key finding**: Execution infrastructure quality (containerization, error handling, metric extraction) has outsized impact — often more than agent logic changes.

### Code Generation Patterns

| Pattern | Agents | Effectiveness |
|---------|--------|--------------|
| Full file regeneration | AIDE, early agents | Wasteful, loses good code sections |
| SEARCH/REPLACE diff patching | MLEvolve | Efficient for refinement, preserves working code |
| Adaptive mode selection (full/diff/multi-agent) | MLEvolve, LoongFlow (Fuse Mode) | Best — matches approach to situation |
| Template-based cold start | MLEvolve (per-category model recommendations) | Reduces wasted early iterations |
| Stepwise multi-agent pipeline | MLEvolve | Better for complex tasks |
| Iterative debug-on-sample then full-run | R&D-Agent | Cost-efficient verification |

**Key finding**: Adaptive code generation that switches modes based on context (exploration vs refinement, simple vs complex) outperforms fixed approaches.

### Search/Exploration Strategies

| Strategy | Agent | Score | Sample Efficiency |
|----------|-------|-------|-------------------|
| Greedy (best-first) | AIDE | 17.1% | Baseline |
| AIDE Greedy (improved infra) | AIRA-dojo | 39.8% | Moderate |
| AIRA Greedy (improved operators) | AIRA-dojo | 45.5% | Moderate |
| MCTS with UCB | ML-Master 1.0 | 29.3% | Better — explores worse branches that lead to improvements |
| AIRA MCTS | AIRA-dojo | 47.0% | Better |
| Evolutionary | PiEvolve | 61.3% | Good with priority decay |
| MCGS (Monte Carlo Graph Search) | MLEvolve | 61.3% | Best — 12h matches 24h agents |
| PES + MAP-Elites + Boltzmann | LoongFlow | 62.7% | Best — 60% fewer calls than OpenEvolve |
| Cost-constrained MCTS | MARS+ | 62.7% | Good — explicitly manages budget |

**Key finding**: The frontier has moved from "which search algorithm" to "how to manage the search" — stagnation detection, diversity maintenance, and time-aware switching matter more than the base algorithm choice.

### Memory Systems Comparison

| Memory Type | Agent | Description | Impact |
|-------------|-------|-------------|--------|
| None | AIDE | Stateless — each node from scratch | Baseline |
| Scoped memory (bounded context) | AIRA-dojo | Recent successes/failures in context | Negligible in their ablation |
| Structured R&D logs | R&D-Agent | Plan → implementation → result tuples | Significant (enables dynamic planning) |
| Experience-driven (BM25+FAISS) | MLEvolve | Full plan/code/metric/label per node, hybrid retrieval | Enables cross-branch fusion |
| Comparative reflective memory | MARS+ | Analyzes solution differences across branches | 63% of lessons from cross-branch transfer |
| 3-tier HCC (L1/L2/L3) | ML-Master 2.0 | Progressive distillation: traces → summaries → wisdom | 22.7% → 72.7% (ablation) |
| Multi-structure fusion | LoongFlow | Evolution tree + MAP-Elites + entropy regulation | Prevents diversity collapse |
| Knowledge graph + wiki | KAPSO | External domain knowledge + experiment traces | Domain-informed exploration |

**Contradiction flagged**: AIRA-dojo found memory operators had "negligible effect" in their ablation, while every other top agent attributes significant performance to memory systems. Possible explanation: AIRA-dojo's memory implementation was simpler (scoped recent context) vs. the sophisticated tiered/retrieval systems in MLEvolve, ML-Master 2.0, and MARS+. Simple memory doesn't help; structured, retrievable memory does.

---

## Contradictions and Divergences

### 1. Memory: Negligible (AIRA-dojo) vs. Critical (everyone else)
- **AIRA-dojo ablation**: Memory operator showed "nearly identical performance with/without"
- **ML-Master 2.0 ablation**: L1-only (no memory hierarchy) = 22.7% vs full HCC = 72.7%
- **MARS+**: 63% of utilized lessons from cross-branch memory transfer
- **Resolution**: The quality and structure of memory matters enormously. Simple "recent context" memory (AIRA-dojo) doesn't help. Tiered, retrievable, and cross-branch memory (ML-Master 2.0, MLEvolve, MARS+) is game-changing. Memory is only as good as its retrieval mechanism.

### 2. Multi-agent count: More agents ≠ better performance
- **Neo**: 36-hour multi-agent → 34.2%
- **MLEvolve**: 3-agent, 12-hour → 61.3%
- **Resolution**: Role quality > agent quantity. Well-defined role separation (planner/executor/reviewer) beats adding more agents with overlapping roles.

### 3. Extended compute: Diminishing returns after 24h
- **AIRA-dojo**: 90 hours → only +6% over 24h (peaked at 53%)
- **Neo**: 36 hours → only 34.2%
- **MLEvolve**: 12 hours → 61.3% (better than most 24h agents)
- **Resolution**: Better search strategy is more valuable than more compute time. The quality of each iteration matters more than the number of iterations.

### 4. MCTS vs Evolutionary: No clear winner
- **MCTS agents** (ML-Master, AIRA-MCTS, MARS+): 29-63%
- **Evolutionary agents** (PiEvolve, LoongFlow, MLEvolve): 61-63%
- **Resolution**: At the frontier, both achieve similar scores. The wrapping infrastructure (memory, reflection, diversity maintenance) matters more than the base search algorithm.

---

# Appendix: Technique Taxonomy

| # | Technique | Category | Source Agent(s) | Validation Status | Performance Delta | Transferability |
|---|-----------|----------|----------------|-------------------|-------------------|-----------------|
| 1 | Tree search in code space | search_strategy | AIDE | multiple_agents_use_it | Baseline (4× vs linear) | High |
| 2 | MCTS with UCB selection | search_strategy | ML-Master 1.0, AIRA-dojo, MARS+ | explicitly_compared | +12-15% absolute vs greedy | High |
| 3 | Evolutionary search | search_strategy | PiEvolve, LoongFlow, MLEvolve | multiple_agents_use_it | +44% absolute vs AIDE | High |
| 4 | Monte Carlo Graph Search (MCGS) | search_strategy | MLEvolve | single_report | 61.3% in 12h | High |
| 5 | MAP-Elites diversity maintenance | search_strategy | LoongFlow | single_report | Prevents premature convergence | High |
| 6 | Entropy-regularized Boltzmann selection | search_strategy | LoongFlow | single_report | Self-adaptive exploration | Medium |
| 7 | Cross-branch fusion | search_strategy | MLEvolve, MARS+ | multiple_agents_use_it | Rescues stalled search; 63% cross-branch transfer | High |
| 8 | Piecewise exploration decay | search_strategy | MLEvolve | single_report | Time-aware explore→exploit | Medium |
| 9 | Automatic stagnation detection | search_strategy | MLEvolve | single_report | Triggers recovery when stuck | High |
| 10 | Cost-constrained MCTS | search_strategy | MARS+ | single_report | Budget-aware planning | High |
| 11 | Priority-based sampling with decay | search_strategy | PiEvolve | single_report | Avoids local optima | Medium |
| 12 | Research/Development phase separation | agent_architecture | R&D-Agent | explicitly_compared | Largest single-component contribution | High |
| 13 | PES (Plan-Execute-Summary) paradigm | agent_architecture | LoongFlow | explicitly_compared | +52% faster (planner), eliminates cycles (summarizer) | Very High |
| 14 | Multi-agent role separation (Planner/Executor/Reviewer) | agent_architecture | MLEvolve, LoongFlow, R&D-Agent | multiple_agents_use_it | Consistent in top performers | High |
| 15 | Multi-model ensemble | agent_architecture | Disarray | single_report | +13% over single-model | Medium |
| 16 | Design-Decompose-Implement pipeline | agent_architecture | MARS+ | single_report | Manages complex codebases | High |
| 17 | Iterative agent loop | agent_architecture | AIBuildAI | single_report | 63.1% overall | Medium |
| 18 | Four-pillar system (Evolve/Learn/Research/Deploy) | agent_architecture | KAPSO | single_report | 50.7% overall | Medium |
| 19 | Hierarchical Cognitive Caching (3-tier) | memory | ML-Master 2.0 | explicitly_compared | 22.7% → 72.7% (ablation) | Very High |
| 20 | Experience-driven memory (BM25+FAISS) | memory | MLEvolve | single_report | Enables cross-branch fusion | High |
| 21 | Comparative reflective memory | memory | MARS+ | single_report | 63% cross-branch lesson transfer | High |
| 22 | Multi-structure fusion memory | memory | LoongFlow | single_report | Evolution tree + MAP-Elites | High |
| 23 | Knowledge graph + domain wiki | memory | KAPSO | single_report | Domain-informed exploration | High |
| 24 | External knowledge retrieval | memory | MLE-STAR, KAPSO | multiple_agents_use_it | Step-function improvement | High |
| 25 | Scoped recent context memory | memory | AIRA-dojo | contradicted | Negligible in ablation | Low |
| 26 | Structured R&D logs (plan→impl→result) | memory | R&D-Agent | single_report | Enables dynamic planning | High |
| 27 | Context compression (>200k → ~70k tokens) | memory | ML-Master 2.0 | single_report | Maintains performance with 65% less context | Very High |
| 28 | Post-iteration structured reflection | evaluation | LoongFlow (Summary), MARS+ | multiple_agents_use_it | Eliminates cyclical errors, 60% efficiency gain | Very High |
| 29 | Think tokens / explicit reasoning prompts | prompting | AIRA-dojo | single_report | 2× completion tokens | Medium |
| 30 | Adaptive prompt complexity | prompting | AIRA-dojo, MLEvolve | multiple_agents_use_it | Matches guidance to task difficulty | High |
| 31 | Dynamic planning (adapts strategy to results) | prompting | R&D-Agent | explicitly_compared | Largest single-component impact in R&D-Agent | High |
| 32 | Cold-start model recommendations | prompting | MLEvolve | single_report | Reduces wasted early iterations | High |
| 33 | Reasoning model backbone (R1, o3) | prompting | ML-Master, AIRA-dojo, R&D-Agent | multiple_agents_use_it | Substantial gains over chat models | High |
| 34 | Execution infrastructure quality | tool_usage | AIRA-dojo | explicitly_compared | +10.7% absolute from infra alone | Very High |
| 35 | Containerized sandboxing | tool_usage | AIRA-dojo, KAPSO | multiple_agents_use_it | Reliability + reproducibility | High |
| 36 | W&B experiment tracking integration | tool_usage | AIRA-dojo | single_report | Better metric extraction | Medium |
| 37 | Git-native experimentation (branch-per-attempt) | tool_usage | KAPSO | single_report | Reproducibility + provenance | High |
| 38 | Web search / research integration | tool_usage | KAPSO | single_report | External information access | Medium |
| 39 | SEARCH/REPLACE diff patching | code_generation | MLEvolve | single_report | Efficient refinement, preserves working code | High |
| 40 | Adaptive code generation mode selection | code_generation | MLEvolve, LoongFlow | multiple_agents_use_it | Matches approach to context | High |
| 41 | Stepwise multi-agent code pipeline | code_generation | MLEvolve | single_report | Better for complex tasks | Medium |
| 42 | Debug-on-sample then full-run | code_generation | R&D-Agent | single_report | Cost-efficient verification | High |
| 43 | Component-wise pipeline optimization | code_generation | MLE-STAR, MARS+ | multiple_agents_use_it | Reduces search space | High |
| 44 | Skill packages (domain knowledge encoding) | code_generation | LoongFlow | single_report | Encodes domain expertise | Medium |
| 45 | Model-mixing (expensive reasoning + cheap coding) | parallelism | R&D-Agent | single_report | Cost reduction | High |
| 46 | Parallel exploration directions | parallelism | ML-Master 2.0, AIRA-dojo | multiple_agents_use_it | Broader search | Medium |
| 47 | Multi-trace execution | parallelism | R&D-Agent | single_report | +10% on High-difficulty | Medium |
| 48 | Generalization gap (9-13% val vs test) | failure_mode | AIRA-dojo | explicitly_compared | Unsolved — agents overfit to proxy metrics | High |
| 49 | Diversity collapse (converging to one solution type) | failure_mode | LoongFlow (solved), AIDE (exhibited) | explicitly_compared | MAP-Elites + entropy regulation solves it | High |
| 50 | Cyclical errors (repeating same mistakes) | failure_mode | LoongFlow (solved), AIDE (exhibited) | explicitly_compared | Post-iteration reflection solves it | Very High |
| 51 | Context saturation (raw traces overwhelming LLM) | failure_mode | ML-Master 2.0 (solved), AIDE (exhibited) | explicitly_compared | HCC tiered compression solves it | Very High |
| 52 | Greedy local optima (stuck on best-so-far) | failure_mode | ML-Master 1.0 (solved), AIDE (exhibited) | explicitly_compared | MCTS/evolutionary solves it | High |
| 53 | Time-aware explore-exploit switching | adaptation | MLEvolve | single_report | Auto-transitions based on remaining budget | High |
| 54 | Cross-task knowledge transfer (L3 cache) | adaptation | ML-Master 2.0 | single_report | Warm-start from similar prior tasks | High |
| 55 | Cascaded retrieval (WSR + PFR + ERA) | adaptation | KAPSO | single_report | Heterogeneous source integration | Medium |

---

## Gaps Identified in Pre-Existing Research

### Gaps in Claude's Analysis (initial_analysis_claude.md)
- **Missing agents**: AIBuildAI (#2 on current leaderboard), CAIR MARS+ (different from MLE-STAR — uses MARS framework). Claude's leaderboard snapshot appears to be from a slightly earlier version.
- **Disarray/LoongFlow categorization**: Claude lists them in the main leaderboard; current leaderboard separates them due to test-set feedback advantage.
- **ML-Master 2.0 HCC details**: Claude mentions ML-Master 2.0 but doesn't detail the three-tier memory hierarchy or ablation results.

### Gaps in Gemini's Analysis (initial_analysis_gemini.md)
- **Shallow on most agents**: Only covers 5 agents at surface level.
- **AIBuildAI mentioned**: But describes "Recursive Self-Improvement Loops" without specifics.
- **"Context-First Integration" / ACE framework for Disarray**: Interesting claim but not independently verified.
- **Missing**: AIRA-dojo, R&D-Agent, ML-Master, MARS+, PiEvolve, KAPSO, all ablation data.

### Gaps in OpenAI's Analysis (initial_analysis_openai.md)
- **Most superficial**: No detailed architecture descriptions, no ablation results, no specific metrics beyond approximate scores.
- **LoongFlow described as "Workflow orchestration"**: Significantly undersells the PES paradigm and evolutionary memory innovations.
- **Missing**: All agents below MLEvolve, all papers, all ablation data, all memory system comparisons.

### Gaps Remaining in This Report
- **Famou-Agent 2.0**: #1 on main leaderboard but closed source — no architectural details available. The v1.0 → v2.0 jump (+20.8%) suggests major innovations we cannot document.
- **AIBuildAI**: Repo exists but limited documentation beyond high-level loop description.
- **Disarray ACE framework**: Only mentioned in Gemini's analysis, not independently verified.
- **PiEvolve internals**: Closed source — only press-release-level information.
- **Thesis internals**: Closed source beyond blog post.
- **Operand, Neo**: No architectural information at all.
- **Ablation data**: Only AIRA-dojo, LoongFlow, ML-Master 2.0, and partially R&D-Agent provide rigorous ablations. Other agents' contributions are inferred from leaderboard correlations, not controlled experiments.

---

## Further Reading (Papers and Resources)

| Resource | URL | Relevance |
|----------|-----|-----------|
| AIDE paper | arXiv:2502.13138 | Baseline methodology, tree search formalization |
| AIRA-dojo paper | arXiv:2507.02554 | Search policy comparison, infrastructure impact, generalization gap |
| LoongFlow paper | arXiv:2512.24077 | PES paradigm, MAP-Elites, sample efficiency |
| R&D-Agent paper | arXiv:2505.14738 | R&D phase separation, dynamic planning |
| ML-Master 1.0 paper | arXiv:2506.16499 | MCTS for ML engineering |
| ML-Master 2.0 paper | arXiv:2601.10402 | Hierarchical Cognitive Caching |
| MARS paper | arXiv:2602.02660 | Cost-constrained MCTS, reflective memory |
| KAPSO paper | arXiv:2601.21526 | Knowledge-grounded synthesis |
| AutoMLGen paper | arXiv:2510.08511 | MLEvolve precursor |
| MLE-Dojo paper | arXiv:2505.07782 | Training environment for ML agents |
| MLE-Bench paper | arXiv:2410.07095 | Original benchmark paper |
| R&D-Agent-Quant paper | arXiv:2505.15155 | Financial application of R&D-Agent |

---

*Report generated March 22, 2026. Data sourced from MLE-Bench GitHub leaderboard, framework repositories, associated academic papers, and three pre-existing analysis documents.*
