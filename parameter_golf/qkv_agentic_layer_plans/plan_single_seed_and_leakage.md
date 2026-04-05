# QKayV Feature Implementation Plan: Single-Seed Result Tracker & Information Leakage Detector

**Date**: 2026-03-24
**Status**: Planning
**Related Research**: Parameter Golf Techniques Audit (2026-03-24), Issue #402 (TTT Information Leakage)

---

## Executive Summary

This plan details two complementary features for QKayV:

1. **Single-Seed Result Tracker**: Distinguishes statistically-validated findings from single-observation claims, preventing unvalidated results from influencing downstream decisions.

2. **Information Leakage Detector**: Formally defines and detects TTT-related information leakage categories, preventing invalid experimental results from entering the knowledge base.

Both features share the experiment metadata substrate and integrate with the dependency graph and failure mode KB. Together they form a **validation layer** that ensures QKayV's recommendations are both statistically sound and methodologically valid.

---

## Part A: Single-Seed Result Tracker

### A1. Data Model

#### Core Schema: `ExperimentResult`

```
ExperimentResult {
  id: UUID
  technique: TechniqueReference          # Links to dependency graph node
  outcome: OutcomeMetric                 # BPB, accuracy, etc.
  seed_count: int                        # Number of seeds run
  seeds: list[float]                     # Individual seed values
  mean: float                            # Arithmetic mean
  std: float                             # Standard deviation
  baseline_mean: float                   # Baseline comparison value
  p_value: float | null                  # vs baseline, if computed
  effect_size: float | null             # Cohen's d or similar
  conditions: ExperimentConditions       # Hardware, data, hyperparameters
  validation_status: ValidationStatus    # Enum
  confidence_interval: tuple[float, float] | null
  metadata: dict                        # Extensible

  # Claim tracking
  claimed_improvement: float | null     # What PR author claimed
  actual_improvement: float | null      # Measured vs baseline
  claim_source: str                     # PR number, issue, manual entry
  claim_verified: bool                  # Has claim been checked?
}

enum ValidationStatus {
  SINGLE_SEED_PENDING    # 1 seed, flagged as preliminary
  MULTI_SEED_PENDING     # >1 seed, validation in progress
  VALIDATED_SIGNIFICANT  # Multi-seed, p < 0.05, meaningful effect size
  VALIDATED_NULL         # Multi-seed, no significant difference
  VALIDATED_NEGATIVE     # Multi-seed, significant regression
  INCONCLUSIVE           # High variance, insufficient power
}
```

#### Claim vs. Evidence Tracking

```
ClaimRecord {
  id: UUID
  experiment_result_id: UUID             # Links to ExperimentResult
  claim_type: enum {
    IMPROVEMENT,       # "X is better than Y"
    EQUIVALENCE,       # "X is comparable to Y"
    SUPERIORITY,       # "X beats all alternatives"
    NULL_EFFECT,       # "X has no effect on Y"
  }
  claimed_delta: float                   # e.g., "-0.015 BPB"
  claimed_confidence: str                # e.g., "p < 0.01"
  evidence_status: enum {
    UNVERIFIED         # No validation yet
    SUPPORTED          # Evidence confirms claim
    CONTRADICTED       # Evidence refutes claim
    PARTIAL            # Evidence partially supports
    RETRACTED          # Original claim withdrawn
  }
  multi_seed_required: bool              # Should this require >1 seed?
  minimum_seeds_for_validation: int     # Typically 3-5
  validation_history: list[ValidationEvent]
}
```

#### Required Fields Specification

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `technique` | TechniqueReference | Yes | Must link to dependency graph |
| `outcome` | OutcomeMetric | Yes | BPB, bits per byte |
| `seed_count` | int | Yes | 1 = single, 3+ = multi |
| `seeds` | list[float] | Yes | Raw per-seed measurements |
| `mean` | float | Yes | Computed from seeds |
| `std` | float | Yes | Computed from seeds |
| `p_value` | float\|null | No | Welch's t-test recommended |
| `effect_size` | float\|null | No | Cohen's d |
| `baseline_mean` | float | Yes | For delta calculation |
| `validation_status` | ValidationStatus | Yes | Derived field |
| `conditions` | ExperimentConditions | Yes | Full provenance |
| `claimed_improvement` | float\|null | No | PR claim if exists |
| `ci_95` | tuple\|null | No | Bootstrap CI |

#### Storage Format

Experiments stored as JSONL documents in `~/.qkv/experiments/`:

```
~/.qkv/experiments/
  experiments.jsonl          # All ExperimentResult records
  claims.jsonl                # ClaimRecord entries
  validation_log.jsonl       # Audit trail
```

### A2. Validation Rules

#### Minimum Seed Count Thresholds

| Claim Type | Minimum Seeds | Rationale |
|------------|---------------|-----------|
| Routine ablation (confirming established technique) | 2 | Detect gross errors |
| Improvement claim over baseline | 3 | p-value meaningful |
| Record-breaking / novel technique | 5 | Strong evidence bar |
| Competition submission | 5+ | Significance at p < 0.01 |

**Hard rule**: Claims in the knowledge base without seed_count >= 3 are labeled `SINGLE_SEED_PENDING` and cannot be marked `VALIDATED_SIGNIFICANT`.

#### Statistical Tests

**Primary**: Welch's t-test (unequal variances assumed)

```python
from scipy import stats

def welch_ttest(experiment_seeds, baseline_seeds, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(experiment_seeds, baseline_seeds,
                                        equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value, "significant": p_value < alpha}
```

**Supplementary**: Bootstrap confidence intervals (1000 iterations)

```python
import numpy as np

def bootstrap_ci(experiment_seeds, baseline_seeds, n_bootstrap=1000, alpha=0.05):
    """95% CI for the difference between experiment and baseline means."""
    delta_samples = []
    for _ in range(n_bootstrap):
        exp_sample = np.random.choice(experiment_seeds, size=len(experiment_seeds), replace=True)
        base_sample = np.random.choice(baseline_seeds, size=len(baseline_seeds), replace=True)
        delta_samples.append(np.mean(exp_sample) - np.mean(base_sample))
    return np.percentile(delta_samples, [alpha/2*100, (1-alpha/2)*100])
```

**Effect size**: Cohen's d

```python
def cohens_d(experiment_seeds, baseline_seeds):
    pooled_std = np.sqrt(((len(experiment_seeds)-1)*np.var(experiment_seeds, ddof=1) +
                          (len(baseline_seeds)-1)*np.var(baseline_seeds, ddof=1)) /
                         (len(experiment_seeds) + len(baseline_seeds) - 2))
    return (np.mean(experiment_seeds) - np.mean(baseline_seeds)) / pooled_std
```

#### What Constitutes "Significant" Improvement

Three conditions must ALL be met:

1. **Statistical**: p-value < 0.05 (Welch's t-test, two-sided)
2. **Practical**: Effect size |d| > 0.3 (small effect minimum) OR absolute delta > 0.001 BPB
3. **Reproducibility**: At least 3 seeds, std < |delta| (means are separated by > 1 std)

**BPB-specific thresholds**:
- Meaningful improvement: delta < -0.001 BPB with p < 0.05
- Competition submission threshold (parameter golf): delta < -0.005 BPB, p < 0.01
- Routine technique adoption: delta < -0.0005 BPB with p < 0.05, effect size > 0.5

#### Variance Thresholds

| Variance (std) | Interpretation | Action |
|----------------|----------------|--------|
| std < 0.001 BPB | Very stable | Accept as validated |
| 0.001 <= std < 0.005 BPB | Normal | Standard validation |
| 0.005 <= std < 0.01 BPB | High variance | Flag, require more seeds |
| std >= 0.01 BPB | Very high | Mark INCONCLUSIVE, investigate |

### A3. Concrete Examples from Parameter Golf

#### Example 1: PR #490 — 1.0891 BPB (Should be flagged: SINGLE_SEED_PENDING)

```
ExperimentResult {
  technique: "Value Residual + Gated Attention + AdamW TTT"
  outcome: 1.0891 BPB
  seed_count: 1
  seeds: [1.0891]
  mean: 1.0891
  std: 0.0  # Cannot compute std from 1 seed
  baseline_mean: 1.1195  # Previous record (PR #528)
  claimed_improvement: -0.0304 BPB
  validation_status: SINGLE_SEED_PENDING
  claim_source: "PR #490"
  conditions: {
    pr_number: 490,
    author: "amaljithkuttamath",
    submitted: "2026-03-22",
    note: "Pending multi-seed validation"
  }
}
```

**QKayV Response**: "PR #490 claims 1.0891 BPB (-0.030 BPB improvement). WARNING: This is a single-seed result. Validation status is SINGLE_SEED_PENDING. Before acting on this result, run 3-5 additional seeds. The apparent improvement could be within natural seed variance."

#### Example 2: PR #442 — AdamW TTT replacing SGD (Should be flagged: MULTI_SEED_PENDING)

```
ExperimentResult {
  technique: "AdamW TTT (replacing SGD TTT)"
  outcome: 1.1027 BPB
  seed_count: 1
  seeds: [1.1027]
  mean: 1.1027
  std: 0.0
  baseline_mean: 1.1214  # PR #473 (SGD TTT baseline)
  claimed_improvement: -0.0187 BPB (AdamW vs SGD for TTT)
  validation_status: SINGLE_SEED_PENDING  # Despite being marked "Single-seed result"
  claim_source: "PR #442"
  note: "Single-seed result pending multi-seed validation"
  conditions: {
    ttt_optimizer: "AdamW",
    ttt_lr: 0.0005,
    ttt_epochs: 10,
    comparison: "SGD TTT"
  }
}
```

**QKayV Response**: "PR #442 claims AdamW TTT outperforms SGD TTT by -0.019 BPB. This is a single-seed result marked SINGLE_SEED_PENDING. Given the magnitude of the claimed effect (-0.019 BPB), it is likely robust, but validation with 3+ seeds is required for confident knowledge base entry."

#### Example 3: PR #528 — 1.1195 BPB (Partially validated, partial concern)

```
ExperimentResult {
  technique: "Full GPTQ + Early QAT + Legal score-first TTT AdamW"
  outcome: 1.1195 BPB
  seed_count: 1
  seeds: [1.1195]
  mean: 1.1195
  std: 0.0
  baseline_mean: 1.1233  # Prior record
  claimed_improvement: -0.0038 BPB
  validation_status: SINGLE_SEED_PENDING
  claim_source: "PR #528"
  note: "Full GPTQ beats GPTQ-lite by ~0.002 BPB but requires more compute"
  conditions: {
    quantization: "GPTQ full (Hessian, Cholesky, 256-sample)",
    qat_timing: "Early QAT 1750 steps",
    ttt_optimizer: "AdamW",
    ttt_epochs: 3,
    unfrozen_params: "4.7M"
  }
}
```

**Analysis**: The claim is "Full GPTQ beats GPTQ-lite by ~0.002 BPB." This is a delta of -0.002 BPB, which is below the typical single-seed variance (std of ~0.002-0.005 BPB for these models). QKayV should flag this as requiring multi-seed validation before the GPTQ vs GPTQ-lite comparison is trusted.

#### Example 4: Early PRs — Single-seed treated as preliminary

```
ExperimentResult {
  technique: "Sliding window evaluation (stride=64)"
  outcome: 1.1925 BPB
  seed_count: 1  # Original submission
  seeds: [1.1925]
  mean: 1.1925
  std: 0.0
  baseline_mean: 1.2014 BPB  # Previous without sliding window
  claimed_improvement: -0.009 BPB
  validation_status: VALIDATED_SIGNIFICANT  # Later confirmed by 50+ reproductions
  conditions: {
    note: "50+ independent reproductions confirm this result"
  }
}
```

**Note**: This result was initially single-seed but became VALIDATED_SIGNIFICANT through community reproduction. The validation mechanism here is informal multi-source reproduction rather than formal multi-seed. QKayV should support both validation paths: (a) same technique, multiple seeds by same team, and (b) same technique, multiple independent implementations.

### A4. Query Interface

#### Q1: "Is this finding statistically validated?"

```python
def query_validation_status(technique_id: str) -> ValidationStatus:
    """
    Returns the validation status and supporting evidence for a technique.
    """
    result = find_experiment(technique_id)
    if not result:
        return {"found": False}

    return {
        "technique": result.technique,
        "validation_status": result.validation_status,
        "seed_count": result.seed_count,
        "mean": result.mean,
        "std": result.std,
        "p_value": result.p_value,
        "effect_size": result.effect_size,
        "ci_95": result.confidence_interval,
        "validation_evidence": get_validation_evidence(result.id),
        "reproductions": count_reproductions(technique_id),
    }
```

**Example response**:
```
Technique: AdamW TTT (replacing SGD TTT)
Validation Status: SINGLE_SEED_PENDING

Seed count: 1 (minimum required: 3)
Mean: 1.1027 BPB
Std: N/A (single seed)

p-value: N/A (requires baseline multi-seed)
Effect size: N/A

Recommendation: Run 3-5 additional seeds before this finding
can be considered validated. Claimed improvement of -0.019 BPB
is large enough to likely replicate, but variance is unknown.
```

#### Q2: "What is the confidence interval on this improvement?"

```python
def query_confidence_interval(technique_id: str, baseline_id: str) -> dict:
    """
    Returns the 95% CI on the improvement delta.
    """
    exp = find_experiment(technique_id)
    base = find_experiment(baseline_id)

    if exp.seed_count < 3 or base.seed_count < 3:
        ci = bootstrap_ci(exp.seeds, base.seeds)
        return {
            "ci_95": ci,
            "method": "bootstrap (low n)",
            "warning": "Small sample size; CI may be wide"
        }
    else:
        ci = bootstrap_ci(exp.seeds, base.seeds, n_bootstrap=10000)
        return {"ci_95": ci, "method": "bootstrap (n>=3)"}
```

**Example response**:
```
Improvement: Full GPTQ vs GPTQ-lite
Point estimate: -0.002 BPB
95% CI: [-0.008, +0.001] BPB

WARNING: CI includes zero. Improvement is not statistically
significant at the 95% level. Single-seed results make
CI unreliable. Run additional seeds to tighten interval.
```

#### Q3: "Which techniques in my stack lack multi-seed validation?"

```python
def query_unvalidated_techniques(stack: list[str]) -> list[dict]:
    """
    Returns all techniques in the given stack that lack multi-seed validation.
    """
    unvalidated = []
    for technique_id in stack:
        result = find_experiment(technique_id)
        if not result:
            unvalidated.append({
                "technique": technique_id,
                "reason": "NOT_FOUND_IN_KB",
                "severity": "HIGH"
            })
        elif result.seed_count < 3:
            unvalidated.append({
                "technique": technique_id,
                "validation_status": result.validation_status,
                "seed_count": result.seed_count,
                "risk": "downstream_decisions_may_rest_on_unvalidated_result"
            })
    return unvalidated
```

**Example response**:
```
Stack validation report for PR #544 base:
- XSA (final 4 layers): VALIDATED_SIGNIFICANT (50+ reproductions)
- BigramHash (8192): VALIDATED_SIGNIFICANT (100+ reproductions)
- Int5 GPTQ: SINGLE_SEED_PENDING (1 seed)
- Score-first AdamW TTT: MULTI_SEED_PENDING (3+ seeds, p=0.03)

Action items:
1. Run 2+ additional seeds for Int5 GPTQ before treating as validated
2. Score-first TTT is nearly validated; confirm p < 0.05 threshold
```

#### Q4: "Should I run more seeds before acting on this result?"

```python
def query_seed_recommendation(technique_id: str) -> dict:
    """
    Returns a recommendation on whether to run more seeds.
    """
    result = find_experiment(technique_id)

    # Calculate the "sigma gap" — how many stds separate means
    if result.seed_count < 2:
        return {
            "recommendation": "RUN_MORE_SEEDS",
            "priority": "HIGH",
            "reason": "Only 1 seed available; variance completely unknown",
            "suggested_seeds": 5
        }

    delta = abs(result.claimed_improvement or 0)
    std = result.std

    if std == 0:
        return {
            "recommendation": "RUN_MORE_SEEDS",
            "priority": "HIGH",
            "reason": "Single seed; cannot estimate variance",
            "suggested_seeds": 5
        }

    sigma_gap = delta / std

    if sigma_gap > 3 and result.seed_count >= 3:
        return {
            "recommendation": "ADEQUATE_SEEDS",
            "priority": "LOW",
            "reason": f"Delta ({delta:.4f}) is {sigma_gap:.1f}x std ({std:.4f})",
            "confidence": "HIGH"
        }
    elif sigma_gap > 2 and result.seed_count >= 3:
        return {
            "recommendation": "ADEQUATE_SEEDS",
            "priority": "MEDIUM",
            "reason": f"Delta ({delta:.4f}) is {sigma_gap:.1f}x std ({std:.4f})",
            "confidence": "MEDIUM"
        }
    else:
        return {
            "recommendation": "RUN_MORE_SEEDS",
            "priority": "HIGH",
            "reason": f"Delta ({delta:.4f}) is only {sigma_gap:.1f}x std ({std:.4f}); vulnerable to variance",
            "suggested_seeds": max(5, result.seed_count + 2)
        }
```

### A5. Integration with Research Loop

#### Pre-Decision Flagging

Before QKayV recommends a technique for inclusion in a stack:

```
IF technique.validation_status == SINGLE_SEED_PENDING:
    FLAG "unvalidated_technique"
    WARN "This technique has only 1 seed. Recommend validation before inclusion."
    SUGGEST follow-up experiments
    ATTENUATE confidence_weight by 0.5

IF technique.validation_status == VALIDATED_SIGNIFICANT:
    SET confidence_weight = 1.0

IF technique.validation_status == VALIDATED_NEGATIVE:
    FLAG "technique_harms_performance"
    EXCLUDE from recommendations unless explicitly requested
```

#### Follow-Up Experiment Suggestion

When a single-seed result is flagged:

```
Your current stack includes "Int5 GPTQ" which has only 1 seed.

Suggested follow-up experiment:
- Run Int5 GPTQ with 4 additional seeds (seed 2-5)
- Keep all other stack components fixed
- Target: determine if the ~0.002 BPB delta from int6 persists

Expected compute: ~40 minutes on 8x H100
Expected outcome: If |delta| > 2*std across 5 seeds, treat as validated
```

#### Confidence-Weighted Recommendations

```
def weighted_technique_score(technique: Technique, base_score: float) -> float:
    validation_multiplier = {
        VALIDATED_SIGNIFICANT: 1.0,
        MULTI_SEED_PENDING: 0.8,
        SINGLE_SEED_PENDING: 0.5,
        INCONCLUSIVE: 0.3,
        VALIDATED_NULL: 0.0,
        VALIDATED_NEGATIVE: -0.5,
    }
    return base_score * validation_multiplier[technique.validation_status]
```

**Example**: Two techniques both score 0.8 on quality grounds. XSA (VALIDATED_SIGNIFICANT, 50+ reproductions) gets weighted score 0.8. Int5 GPTQ (SINGLE_SEED_PENDING) gets weighted score 0.4. QKayV would recommend XSA with higher confidence.

---

## Part B: Information Leakage Detector

### B1. Taxonomy of Leakage (from Issue #402)

#### Category 1: Invalid (Direct Data Leakage)

**Definition**: Training or adaptation directly on validation/test data.

**Examples**:
- Running gradient descent on validation set tokens
- Updating model weights using validation loss
- Using validation data in any learned adaptation step

**Code pattern (INVALID)**:
```python
# INVALID — training on validation data
model.train()
for batch in validation_dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # Gradient updates on val data
    optimizer.step()
```

#### Category 2: Gray Zone (Token-Stream TTT)

**Definition**: Score-first TTT where the token stream is not properly document-isolated, causing information bleed across document boundaries during adaptation.

**The Problem at Scale**: When processing a continuous token stream:
- Document A ends at token 1000, Document B starts at token 1001
- TTT adaptation at token 1001 has already seen tokens from Document A
- This creates implicit leakage: the model adapts to Document B using information from Document A's validation tokens

**When It's Gray**:
- Processing documents in a merged token stream (not document-by-document)
- Not using `torch.inference_mode()` during scoring phases
- Not isolating each document's scoring from subsequent documents

**Code pattern (GRAY)**:
```python
# GRAY — token stream TTT without document isolation
for token_batch in merged_validation_stream:
    with torch.inference_mode():
        scores = model(token_batch)  # Each token scored under inf_mode
    # But adaptation happens across document boundaries
    for doc_tokens in split_by_document(token_batch):
        adapt(doc_tokens)  # Uses boundary info from adjacent docs
```

#### Category 3: Valid (Document-Independent Score-First TTT)

**Definition**: Each document is scored and adapted in complete isolation, under `torch.inference_mode()`, before any weight updates occur.

**Key Invariant**: Every token must be scored under `torch.inference_mode()` BEFORE any weight update.

**Valid code pattern**:
```python
# VALID — document-independent TTT
model.eval()
with torch.inference_mode():
    for doc in validation_documents:  # Each doc in isolation
        # Phase 1: Score all tokens (read-only)
        doc_scores = []
        for i in range(0, len(doc), seq_len):
            chunk = doc[i:i+seq_len]
            with torch.no_grad():  # Explicit inference mode
                scores = model(chunk)
            doc_scores.append(scores)

        # Phase 2: Adapt (only after ALL scoring complete)
        model.train()
        for epoch in range(3):
            for chunk in chunks_of(doc_scores):
                loss = compute_ttt_loss(chunk)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

**Additional Validators**:
- [ ] Documents processed in isolation (not merged streams)
- [ ] `torch.inference_mode()` or `torch.no_grad()` active during scoring
- [ ] No weight updates during scoring phase
- [ ] Adaptation uses frozen embeddings OR embeddings are excluded from TTT loss

### B2. Detection Patterns

#### Pattern 1: Gradient Signal on Validation Data

```python
def detect_gradients_on_validation(log: ExperimentLog) -> list[LeakageViolation]:
    """
    Scan experiment logs for gradient operations during validation phases.
    """
    violations = []

    for step in log.steps:
        if step.phase == "validation" or step.phase == "ttt":
            if step.gradients_computed:
                violations.append({
                    "type": "CATEGORY_1",
                    "severity": "CRITICAL",
                    "step": step.number,
                    "description": f"Gradient computed on {step.phase} data",
                    "details": step.gradients_shape
                })

        if step.phase == "ttt_scoring":
            if step.weights_updated:
                violations.append({
                    "type": "CATEGORY_2_OR_3",
                    "severity": "CRITICAL",
                    "step": step.number,
                    "description": "Weight update during TTT scoring phase",
                    "details": "Weights should only update in TTT adaptation phase, after all scoring"
                })

    return violations
```

#### Pattern 2: Document Isolation Check

```python
def detect_document_isolation_failure(log: ExperimentLog) -> list[LeakageViolation]:
    """
    Check if documents are being processed in isolation during TTT.
    """
    violations = []

    for step in log.steps:
        if step.phase == "ttt_scoring":
            if step.document_boundaries_respected == False:
                violations.append({
                    "type": "CATEGORY_2",
                    "severity": "HIGH",
                    "step": step.number,
                    "description": "Token stream processed without document boundaries",
                    "details": "TTT scoring must isolate each document to prevent cross-document leakage"
                })

            if step.inference_mode_active == False:
                violations.append({
                    "type": "CATEGORY_2_OR_3",
                    "severity": "CRITICAL",
                    "step": step.number,
                    "description": "Scoring without torch.inference_mode()",
                    "details": "All scoring must occur under inference_mode() before any weight updates"
                })

    return violations
```

#### Pattern 3: TTT Phase Ordering

```python
def detect_ttt_phase_ordering(log: ExperimentLog) -> list[LeakageViolation]:
    """
    Verify TTT follows valid sequence: score → adapt (never score → update → score again).
    """
    violations = []

    ttt_phases = [s for s in log.steps if s.phase.startswith("ttt")]

    for i, phase in enumerate(ttt_phases):
        if phase.phase == "ttt_scoring":
            # Check that NO weight update has happened since last scoring
            if phase.weights_modified_since_last_score:
                violations.append({
                    "type": "CATEGORY_2_OR_3",
                    "severity": "CRITICAL",
                    "step": phase.number,
                    "description": "Weight update between scoring and adaptation within same TTT pass",
                    "details": f"Modified params: {phase.modified_param_names}"
                })

        if phase.phase == "ttt_adaptation":
            # Verify scoring happened BEFORE this adaptation
            prior_phases = ttt_phases[:i]
            scoring_done = any(p.phase == "ttt_scoring" for p in prior_phases)
            if not scoring_done:
                violations.append({
                    "type": "CATEGORY_1",
                    "severity": "CRITICAL",
                    "step": phase.number,
                    "description": "TTT adaptation without prior scoring phase",
                    "details": "Must score all documents before any adaptation"
                })

    return violations
```

#### Pattern 4: Learned Parameter Updates on Validation

```python
def detect_learned_param_updates_on_val(log: ExperimentLog) -> list[LeakageViolation]:
    """
    Check if learned components (embeddings, layernorms) are being updated during TTT on val data.
    """
    violations = []

    for step in log.steps:
        if step.phase in ["ttt_adaptation", "validation"]:
            for param_name in step.updated_params:
                if any(learned in param_name for learned in ["embed", "embedding", "layernorm", "ln"]):
                    if step.phase == "ttt_adaptation":
                        # This is the "freeze embeddings" rule
                        violations.append({
                            "type": "CATEGORY_3_BORDERLINE",
                            "severity": "MEDIUM",
                            "step": step.number,
                            "description": f"Embedding/LayerNorm updated during TTT adaptation",
                            "details": "Freezing embeddings is recommended to prevent catastrophic forgetting",
                            "recommendation": "Consider adding embed to frozen_params list"
                        })

    return violations
```

### B3. Known Violations from Parameter Golf

#### PR #532 — Withdrawn (Category 1 Violation)

```
LeakageReport {
  pr_number: 532,
  author: "NotADevIAmaMeatPopsicle",
  claimed_bpb: 1.0487,
  actual_bpb: "DQ'd (withdrawn)",
  violation_type: "TTT_COMPLIANCE",
  category: "CATEGORY_1",
  severity: "CRITICAL",
  description: "TTT implementation did not meet compliance standards",
  details: "PR closed for TTT compliance; compression pipeline itself is valid",
  valid_components: ["Codebook + Huffman compression pipeline"],
  lesson: "Novel compression techniques can survive disqualification; TTT must be reimplemented compliantly"
}
```

**Note**: The compression pipeline (codebook + Huffman) was technically valid and achieved 21% better compression than int6+zstd. The TTT violation caused disqualification, but this is separable.

#### 9 PRs Flagged in Issue #402

Issue #402 documents 9 PRs with TTT violations:

```
LeakageIssue {
  issue_number: 402,
  title: "TTT Information Leakage",
  reporter: "leloykun",
  affected_prs: 9,
  closed_prs: 1,
  status: "OPEN",
  key_rule: "Every token must be scored under torch.inference_mode() BEFORE any weight update",
  categories: {
    CATEGORY_1_INVALID: "Train on validation data directly",
    CATEGORY_2_GRAY: "Token-stream TTT — problematic at scale",
    CATEGORY_3_VALID: "Document-independent score-first TTT"
  }
}
```

#### Adapt-First TTT — Explicitly Labeled as Leakage

```
ViolationType {
  name: "Adapt-First TTT",
  category: "CATEGORY_1",
  severity: "CRITICAL",
  description: "Adapting on validation data BEFORE scoring, then scoring adapted model on same data",
  example_pattern: """
    # INVALID
    model.train()
    for epoch in range(10):
        for doc in val_docs:
            adapt(doc)  # Update weights FIRST
    model.eval()
    with torch.inference_mode():
        for doc in val_docs:
            score(doc)  # Score the adapted model on same data
  """,
  correct_pattern: """
    # VALID
    model.eval()
    with torch.inference_mode():
        for doc in val_docs:
            score(doc)  # Score FIRST under inference_mode
    model.train()
    for epoch in range(3):
        for doc in val_docs:
            adapt(doc)  # THEN adapt (no leakage since scoring was under inf_mode)
  """
}
```

### B4. Query Interface

#### Q1: "Does this training setup introduce information leakage?"

```python
def query_leakage_assessment(setup: ExperimentSetup) -> LeakageAssessment:
    """
    Analyzes an experimental setup for potential information leakage.
    """
    violations = []

    # Check 1: Is validation data in training loop?
    if setup.validation_in_training_loop:
        violations.append({
            "check": "validation_in_training_loop",
            "status": "VIOLATION",
            "category": "CATEGORY_1",
            "message": "Validation data is processed in the same loop as training data"
        })

    # Check 2: Is TTT implemented?
    if setup.has_ttt:
        # Check phase ordering
        if not setup.scoring_before_adaptation:
            violations.append({
                "check": "ttt_phase_ordering",
                "status": "VIOLATION",
                "category": "CATEGORY_1",
                "message": "TTT adaptation occurs before scoring — this is adapt-first leakage"
            })

        # Check inference mode during scoring
        if not setup.inference_mode_during_scoring:
            violations.append({
                "check": "inference_mode_during_scoring",
                "status": "VIOLATION",
                "category": "CATEGORY_2_OR_3",
                "message": "Scoring is not under torch.inference_mode()"
            })

        # Check document isolation
        if not setup.documents_isolated:
            violations.append({
                "check": "document_isolation",
                "status": "WARNING",
                "category": "CATEGORY_2",
                "message": "Documents are not processed in isolation — potential token-stream leakage"
            })

    # Check 3: Frozen parameters
    if setup.ttt_updates_embeddings:
        violations.append({
            "check": "ttt_embedding_updates",
            "status": "WARNING",
            "category": "CATEGORY_3_BORDERLINE",
            "message": "Embeddings are being updated during TTT — risk of catastrophic forgetting"
        })

    return {
        "is_clean": len(violations) == 0,
        "violations": violations,
        "overall_category": categorize_overall(violations),
        "recommendation": generate_recommendation(violations)
    }
```

**Example response**:
```
Leakage Assessment for: PR #532 (before disqualification)

Status: CLEAN with CONCERNS
Category: CATEGORY_3_BORDERLINE

Violations found:
1. [WARNING] Documents not fully isolated during TTT scoring
   - TTT adaptation phase updates weights after scoring
   - Embeddings NOT frozen during adaptation

Recommendations:
1. Implement document isolation (process each doc separately under inference_mode)
2. Freeze embedding parameters during TTT adaptation
3. Verify all scoring happens BEFORE any weight updates

Note: The codebook + Huffman compression pipeline is independent
of the TTT implementation and would remain valid if TTT were
reimplemented compliantly.
```

#### Q2: "Is this TTT implementation valid under Category 3?"

```python
def query_ttt_compliance(code_or_log: ExperimentLog) -> TTTComplianceReport:
    """
    Validates a TTT implementation against Category 3 requirements.
    """
    checks = {
        "inference_mode_during_scoring": False,
        "documents_isolated": False,
        "scoring_before_adaptation": False,
        "no_weight_updates_during_scoring": False,
        "embeddings_frozen_or_excluded": False,
    }

    # Analyze the log/code
    analysis = analyze_ttt_implementation(code_or_log)

    checks["inference_mode_during_scoring"] = analysis.has_inference_mode
    checks["documents_isolated"] = analysis.documents_processed_separately
    checks["scoring_before_adaptation"] = analysis.score_called_before_adapt
    checks["no_weight_updates_during_scoring"] = not analysis.has_updates_in_score_phase
    checks["embeddings_frozen_or_excluded"] = analysis.embeddings_excluded_from_ttt_loss

    all_pass = all(checks.values())
    category = "CATEGORY_3_VALID" if all_pass else "CATEGORY_2_GRAY" if checks["inference_mode_during_scoring"] else "CATEGORY_1_INVALID"

    return {
        "compliant": all_pass,
        "category": category,
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "recommendation": "COMPLIANT" if all_pass else "FIX_REQUIRED"
    }
```

**Example response**:
```
TTT Compliance Report for: PR #528 (Legal score-first TTT AdamW)

Status: COMPLIANT
Category: CATEGORY_3_VALID

Checks:
✓ torch.inference_mode() active during scoring
✓ Documents processed in isolation (131K-token chunks)
✓ All scoring occurs before any weight updates
✓ No weight updates during scoring phase
✓ Embeddings frozen during TTT adaptation

Conclusion: PR #528 implements valid document-independent
score-first TTT. The -0.010 to -0.020 BPB improvement is
legitimately obtained.
```

#### Q3: "Which aspects of my setup could be considered leakage?"

```python
def query_leakage_risk_factors(setup: ExperimentSetup) -> list[RiskFactor]:
    """
    Identifies specific aspects of the setup that could be considered leakage.
    """
    risk_factors = []

    # Check each risk dimension
    risk_factors.extend(check_validation_data_access(setup))
    risk_factors.extend(check_ttt_implementation(setup))
    risk_factors.extend(check_training_loop_interaction(setup))
    risk_factors.extend(check_parameter_update_scope(setup))

    # Sort by severity
    risk_factors.sort(key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x.severity])

    return risk_factors
```

### B5. Prevention vs. Detection

#### Prevention (Pre-Experiment Validation)

```python
class LeakagePreventionValidator:
    """
    Validates experimental setup BEFORE running to prevent wasted compute.
    """

    def validate_setup(self, setup: ExperimentSetup) -> ValidationResult:
        """
        Check setup for leakage risk before experiment runs.
        """
        issues = []

        # Rule 1: No validation data in training
        if setup.uses_validation_in_training:
            issues.append(Issue(
                severity="CRITICAL",
                category="CATEGORY_1",
                message="Validation data cannot be in training loop",
                fix="Separate validation dataloader from training dataloader"
            ))

        # Rule 2: TTT must follow score-first order
        if setup.has_ttt:
            if not setup.ttt_validates_before_adapting:
                issues.append(Issue(
                    severity="CRITICAL",
                    category="CATEGORY_1",
                    message="TTT must score before any adaptation",
                    fix="Reorder TTT phases: score ALL tokens under inference_mode, THEN adapt"
                ))

        # Rule 3: Document isolation for score-first TTT
        if setup.ttt_type == "score_first":
            if not setup.isolates_documents:
                issues.append(Issue(
                    severity="HIGH",
                    category="CATEGORY_2",
                    message="Score-first TTT should isolate documents",
                    fix="Process each document separately under inference_mode before adaptation"
                ))

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            compute_saved_if_valid=setup.estimated_compute_cost if issues else 0
        )
```

**Value of prevention**: A typical parameter golf training run costs ~10 minutes on 8x H100. Catching a leakage issue before running saves this compute.

#### Detection (Post-Hoc Log Analysis)

```python
class LeakageDetector:
    """
    Analyzes experiment logs post-hoc to identify leakage violations.
    """

    def analyze_log(self, log: ExperimentLog) -> LeakageReport:
        """
        Scan log for leakage patterns after experiment completes.
        """
        violations = []

        # Pattern 1: Gradients on validation
        violations.extend(self.detect_gradients_on_validation(log))

        # Pattern 2: Document isolation failures
        violations.extend(self.detect_document_isolation_failure(log))

        # Pattern 3: Phase ordering violations
        violations.extend(self.detect_ttt_phase_ordering(log))

        # Pattern 4: Embedding updates during TTT
        violations.extend(self.detect_embedding_updates(log))

        return LeakageReport(
            experiment_id=log.experiment_id,
            clean=len(violations) == 0,
            violations=violations,
            category=self.categorize(violations),
            impact_assessment=self.assess_impact(violations)
        )

    def assess_impact(self, violations: list[Violation]) -> ImpactAssessment:
        """
        Determine how severely violations affect result validity.
        """
        if not violations:
            return ImpactAssessment(
                validity="VALID",
                confidence="HIGH",
                note="No leakage detected"
            )

        critical = [v for v in violations if v.severity == "CRITICAL"]
        high = [v for v in violations if v.severity == "HIGH"]

        if critical:
            return ImpactAssessment(
                validity="INVALID",
                confidence="CONFIDENCE_LOST",
                note=f"{len(critical)} CRITICAL violations — results are compromised"
            )
        elif high:
            return ImpactAssessment(
                validity="DEGRADED",
                confidence="MEDIUM",
                note=f"{len(high)} HIGH severity concerns — interpret results carefully"
            )
        else:
            return ImpactAssessment(
                validity="VALID_WITH_CONCERNS",
                confidence="LOW",
                note="Minor violations detected; results likely valid but should be reviewed"
            )
```

---

## Part C: Combined Implementation

### C1. Shared Infrastructure

Both features require and share the following infrastructure:

#### Experiment Metadata Store

```
~/.qkv/
  experiments/
    experiments.jsonl      # ExperimentResult records
    claims.jsonl           # ClaimRecord entries
    validation_log.jsonl   # Audit trail
  leakage/
    violation_signatures/  # Known violation patterns
    ttt_patterns/          # Valid/invalid TTT patterns
  kb/
    techniques.json       # Technique registry
    dependency_graph.json # Technique dependencies
    failure_modes.kb       # Failure mode knowledge base
```

#### Shared Data Structures

```
SharedExperimentMetadata {
  experiment_id: UUID
  pr_number: int | null
  source_url: str | null
  technique_tags: list[str]
  outcome_type: enum { BPB, LOSS, ACCURACY, ... }
  conditions: ExperimentConditions
  created_at: datetime
  updated_at: datetime
  provenance: ProvenanceInfo  # Who/when/how recorded
}
```

Both Single-Seed Tracker and Leakage Detector extend this base schema with their specific fields.

#### Integration with Dependency Graph

```
DependencyGraph Integration:
- ExperimentResult.technique links to dependency graph nodes
- Validation status propagates: if technique A depends on B, and B is SINGLE_SEED_PENDING, A's confidence is capped at B's confidence
- Leakage violations on a technique affect all downstream techniques
```

#### Integration with Failure Mode KB

```
FailureModeKB Integration:
- Each leakage violation becomes a failure mode entry
- Detection patterns are indexed by violation type
- "Adapt-first TTT" failure mode links to PRs that exhibited it
- Validation requirements per technique link to single-seed tracking
```

### C2. Implementation Roadmap

#### Phase 1: Metadata Schema + Manual Tracking (Weeks 1-2)

**Goal**: Establish the data model and allow manual experiment entry.

**Deliverables**:
1. `ExperimentResult` schema with all fields from A1
2. `ClaimRecord` schema for claim vs. evidence tracking
3. JSONL storage format with `~/.qkv/experiments/` directory
4. CLI commands for manual experiment entry:
   - `qkv exp add --technique X --outcome 1.123 --seed-count 3 --seeds 1.123,1.124,1.122`
   - `qkv exp add --technique X --claim "improves by 0.01 BPB" --pr 490`
5. Basic query interface:
   - `qkv exp status X` — returns validation status
   - `qkv exp validate X --baseline Y` — runs statistical tests

**Milestone**: User can manually enter experiments and query their validation status.

#### Phase 2: Automated Extraction + Leakage Detection (Weeks 3-4)

**Goal**: Parse experiment logs automatically and detect leakage patterns.

**Deliverables**:
1. Log parser for parameter golf training scripts (detect phases, step types)
2. Automated extraction from PR metadata:
   - Parse PR descriptions for claimed BPB values
   - Extract seed counts from experiment configurations
   - Link PRs to technique taxonomy
3. Leakage detector implementation:
   - Pattern matchers for CATEGORY_1, CATEGORY_2, CATEGORY_3
   - TTT phase ordering validator
   - Document isolation checker
4. Query interface expansion:
   - `qkv leakage scan --pr 532` — assess PR for leakage
   - `qkv leakage check-setup --config config.yaml` — pre-experiment validation
5. Integration with experiment metadata:
   - Leakage status attached to ExperimentResult
   - Violations logged to `~/.qkv/leakage/violations/`

**Milestone**: QKayV can automatically parse a PR's training script and report leakage status.

#### Phase 3: Proactive Validation Planning + Integrated Recommendations (Weeks 5-6)

**Goal**: Actively suggest follow-up experiments and integrate validation/leakage into recommendations.

**Deliverables**:
1. Seed recommendation engine:
   - Given a technique with N seeds, recommend additional seeds
   - Based on effect size, variance, and desired confidence
2. Follow-up experiment planner:
   - `qkv exp plan-validation --technique X` — generates validation experiment spec
   - Estimates compute cost
   - Specifies exact seeds to run
3. Confidence-weighted recommendations:
   - Integrate validation status into technique scoring
   - Integrate leakage status into recommendation eligibility
4. Knowledge base integration:
   - Validated techniques gain higher recommendation weight
   - Leaky techniques are flagged or excluded
   - Failure mode KB enriched with validation findings
5. Reporting:
   - `qkv report validation-gaps` — lists all unvalidated techniques in KB
   - `qkv report leakage-summary` — summarizes leakage findings across all PRs

**Milestone**: QKayV proactively guides users toward validated techniques and flags unvalidated ones.

#### Phase 1-3 Summary Table

| Phase | Duration | Core Capability | Key Deliverables |
|-------|----------|-----------------|------------------|
| 1 | Weeks 1-2 | Metadata schema + manual tracking | JSONL store, CLI entry, basic queries |
| 2 | Weeks 3-4 | Automated extraction + leakage detection | Log parser, leakage detector, PR scanner |
| 3 | Weeks 5-6 | Proactive validation + integrated recommendations | Seed planner, confidence-weighted scoring |

### Implementation Notes

**Technology**: Python with Pydantic for schema validation, JSONL for storage (simple, git-friendly), scipy for statistics.

**Priority ordering within phases**:
1. Schema and storage (foundation)
2. Leakage detection (prevents invalid results from entering KB)
3. Statistical validation (ensures results are reproducible)
4. Query interface (user-facing value)
5. Proactive planning (advanced feature)

**Testing strategy**: Use parameter golf PRs as ground truth — PR #532 (withdrawn), PR #490 (pending validation), PR #528 (valid TTT), PR #442 (single-seed).

**Known edge cases**:
- Single-seed results with very large claimed effects (|delta| >> std) — treat as likely valid but flag anyway
- Multi-source reproduction as validation substitute — count reproductions in addition to seeds
- Gray-zone TTT at small scale — document the scale dependency

---

## Appendix: Reference Tables

### A. Validation Status Transitions

```
SINGLE_SEED_PENDING
  ├── Run 2+ seeds, all similar → MULTI_SEED_PENDING
  ├── Run 2+ seeds, high variance → INCONCLUSIVE
  └── Run 1 seed, widely reproduced → VALIDATED_SIGNIFICANT (via reproduction)

MULTI_SEED_PENDING
  ├── p < 0.05, effect size adequate → VALIDATED_SIGNIFICANT
  ├── p >= 0.05 → VALIDATED_NULL
  ├── p < 0.05, effect size inadequate → INCONCLUSIVE
  └── High variance, underpowered → INCONCLUSIVE

VALIDATED_SIGNIFICANT
  └── Later contradicted by better study → VALIDATED_NULL (or regress)

INCONCLUSIVE
  ├── More seeds resolve → VALIDATED_SIGNIFICANT or VALIDATED_NULL
  └── More seeds keep inconclusive → INCONCLUSIVE (and flag as hard case)
```

### B. Leakage Category Summary

| Category | Name | Severity | Description |
|----------|------|----------|-------------|
| 1 | Invalid | CRITICAL | Train/direct adapt on validation data |
| 2 | Gray | HIGH | Token-stream TTT without document isolation |
| 3 | Valid | — | Document-independent score-first TTT |

### C. TTT Validity Checklist

For any TTT implementation, verify ALL of:

- [ ] `torch.inference_mode()` OR `torch.no_grad()` is active during ALL scoring
- [ ] Every token is scored BEFORE any weight update in the TTT process
- [ ] Documents are processed in isolation (not merged token streams)
- [ ] Scoring phase and adaptation phase are strictly ordered (score → adapt, never adapt → score)
- [ ] Embeddings are frozen OR verified safe from catastrophic forgetting
- [ ] No gradient computation on validation data

### D. Parameter Golf PR Validation Status Reference

| PR | BPB | Technique | Validation Status | Notes |
|----|-----|-----------|-------------------|-------|
| #70 | 1.1659 | Int6+MLP3x+sliding | VALIDATED_SIGNIFICANT | 100+ reproductions |
| #164 | 1.1524 | OrthoInit+BigramHash+SmearGate | VALIDATED_SIGNIFICANT | Systematic stack |
| #198 | 1.1318 | 11L+WD=0.04+SWA | VALIDATED_SIGNIFICANT | Community standard |
| #287 | 1.1271 | XSA+EMA | VALIDATED_SIGNIFICANT | Multiple reproductions |
| #315 | 1.1248 | Partial RoPE+LN Scale | VALIDATED_SIGNIFICANT | Dead code issue noted |
| #379 | 1.1257 | GPTQ-lite | VALIDATED_SIGNIFICANT | 5+ reproductions |
| #442 | 1.1027 | AdamW TTT (vs SGD) | SINGLE_SEED_PENDING | p < 0.01 claimed, needs seeds |
| #466 | 1.1354 | BigramHash 12288 | VALIDATED_SIGNIFICANT | 3-seed mean |
| #473 | 1.1214 | Legal score-first TTT | VALIDATED_SIGNIFICANT | Multiple validations |
| #490 | 1.0891 | Value Residual+AdamW TTT | SINGLE_SEED_PENDING | Withdrawn from record? |
| #507 | 1.1558 | Catalytic Residuals | MULTI_SEED_PENDING | Novel, needs confirmation |
| #528 | 1.1195 | Full GPTQ+TTT | SINGLE_SEED_PENDING | Single seed |
| #532 | DQ | Codebook+Huffman | CATEGORY_1 VIOLATION | TTT non-compliant |
| #538 | 1.1511 | FP8+Arithmetic coding | MULTI_SEED_PENDING | Novel |
| #544 | 1.1179 | Int5 GPTQ+TTT | SINGLE_SEED_PENDING | Single seed |
