# QKayV Feature Implementation Plan v2: Single-Seed Result Tracker & Information Leakage Detector

**Date**: 2026-03-24
**Status**: Planning (v2 — addresses critique findings)
**Related Research**: Parameter Golf Techniques Audit (2026-03-24), Issue #402 (TTT Information Leakage)
**Supersedes**: plan_single_seed_and_leakage.md

---

## Executive Summary

This plan details two complementary features for QKayV:

1. **Single-Seed Result Tracker**: Distinguishes statistically-validated findings from single-observation claims, preventing unvalidated results from influencing downstream decisions.

2. **Information Leakage Detector**: Formally defines and detects TTT-related information leakage categories, flagging invalid experimental results before they enter the knowledge base.

Both features share the experiment metadata substrate and integrate with the dependency graph and failure mode KB. Together they form a **validation layer** that ensures QKayV's recommendations are both statistically sound and methodologically valid.

This v2 plan addresses all critical gaps identified in the critique.

---

## Part A: Single-Seed Result Tracker

### A1. Data Model

#### Core Schema: `ExperimentResult`

```python
class ExperimentResult(BaseModel):
    id: UUID
    technique: TechniqueReference          # Links to dependency graph node
    outcome: OutcomeMetric                 # BPB, accuracy, etc.
    seed_count: int                        # Number of seeds run
    seeds: list[float]                     # Individual seed values
    mean: float                            # Arithmetic mean
    std: float                             # Standard deviation (population)
    baseline_id: UUID | None              # Reference to baseline experiment
    baseline_mean: float                   # Baseline comparison value
    baseline_seed_count: int | None        # For confidence weighting
    p_value: float | None                  # vs baseline, if computed
    effect_size: float | None             # Cohen's d or similar
    conditions: ExperimentConditions        # Hardware, data, hyperparameters
    validation_status: ValidationStatus    # Enum
    confidence_interval: tuple[float, float] | None
    leakage_status: LeakageStatus          # From Part B
    metadata: dict                         # Extensible
    created_at: datetime
    updated_at: datetime

    # Claim tracking
    claimed_improvement: float | None      # What PR author claimed
    actual_improvement: float | None       # Measured vs baseline
    claim_source: str                     # PR number, issue, manual entry
    claim_verified: bool                   # Has claim been checked?

    # Revalidation
    superseded_by: UUID | None             # If a newer experiment supersedes this
    invalidation_reason: str | None       # Why this was superseded
```

#### ValidationStatus Enum (FIXED: Added MIXED_SIGNALS)

```python
class ValidationStatus(str, Enum):
    SINGLE_SEED_PENDING     = "SINGLE_SEED_PENDING"     # 1 seed, flagged preliminary
    MULTI_SEED_PENDING     = "MULTI_SEED_PENDING"      # >1 seed, validation in progress
    MIXED_SIGNALS          = "MIXED_SIGNALS"           # Some seeds positive, some negative
    VALIDATED_SIGNIFICANT  = "VALIDATED_SIGNIFICANT"  # Multi-seed, p < 0.05, meaningful effect
    VALIDATED_NULL         = "VALIDATED_NULL"         # Multi-seed, no significant difference
    VALIDATED_NEGATIVE     = "VALIDATED_NEGATIVE"     # Multi-seed, significant regression
    INCONCLUSIVE            = "INCONCLUSIVE"          # High variance, insufficient power
    RETRACTED               = "RETRACTED"             # Superseded by conflicting evidence
```

**MIXED_SIGNALS Definition**: When N seeds are run (N >= 3) and the sign of the improvement is inconsistent across seeds (e.g., 2/3 show improvement, 1/3 shows regression), the result is MIXED_SIGNALS rather than INVALID. This is distinct from INCONCLUSIVE (which applies when variance is too high to detect any signal).

**MIXED_SIGNALS Handling**:
- Report as "X/N seeds support claimed improvement"
- The burden of proof shifts: a MIXED_SIGNALS technique cannot be VALIDATED_SIGNIFICANT unless subsequent investigation explains the heterogeneity
- Recommend investigating seed-level conditions (different hardware batches, data splits, etc.)

#### Claim vs. Evidence Tracking

```python
class ClaimRecord(BaseModel):
    id: UUID
    experiment_result_id: UUID             # Links to ExperimentResult
    claim_type: ClaimType                  # IMPROVEMENT, EQUIVALENCE, SUPERIORITY, NULL_EFFECT
    claimed_delta: float                    # e.g., "-0.015 BPB"
    claimed_confidence: str                 # e.g., "p < 0.01"
    evidence_status: EvidenceStatus        # UNVERIFIED, SUPPORTED, CONTRADICTED, PARTIAL, RETRACTED
    multi_seed_required: bool               # Should this require >1 seed?
    minimum_seeds_for_validation: int       # Typically 3-5
    validation_history: list[ValidationEvent]

class EvidenceStatus(str, Enum):
    UNVERIFIED   = "UNVERIFIED"
    SUPPORTED    = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    PARTIAL     = "PARTIAL"
    RETRACTED   = "RETRACTED"

class ValidationEvent(BaseModel):
    timestamp: datetime
    event_type: str                        # "seed_added", "status_change", "leakage_found", etc.
    description: str
    actor: str                              # "user", "system", "reproduction"
    evidence: dict                          # Flexible evidence payload
```

#### ReproductionRecord Schema (NEW: Addresses Critique A5)

```python
class ReproductionRecord(BaseModel):
    id: UUID
    technique_id: UUID                     # Technique being reproduced
    original_experiment_id: UUID            # The experiment being reproduced
    reproducer: str                        # Team or individual name
    reproduction_conditions: ExperimentConditions
    result_bpb: float | None
    result_match: bool                     # Does it match original within noise?
    delta_from_original: float | None      # Actual difference
    hardware: str                          # e.g., "8x H100", "A100-80GB"
    software_stack: str                    # Relevant versions
    notes: str
    timestamp: datetime
    confidence: ConfidenceLevel             # HIGH, MEDIUM, LOW based on conditions match
```

**What counts as a reproduction**:
- Same technique evaluated on same or comparable hardware
- Same data split (or validated against comparable validation set)
- Result within 2 standard deviations of original is considered a successful reproduction
- Different teams reproducing independently each count as separate ReproductionRecords

#### Storage Format (FIXED: SQLite instead of JSONL — Addresses Critique A6)

**Storage Architecture**:

```
~/.qkv/
  experiments.db              # SQLite database (see schema below)
  leakage/
    patterns/                 # Known violation pattern signatures
    reports/                  # Generated leakage reports (JSONL, small volume)
  kb/
    techniques.json
    dependency_graph.json
    failure_modes.kb
```

**SQLite Schema**:

```sql
-- Core experiment table with indexed columns for common queries
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,                  -- UUID as text
    technique_id TEXT NOT NULL,
    outcome REAL NOT NULL,
    outcome_type TEXT NOT NULL,            -- 'BPB', 'LOSS', 'ACCURACY'
    seed_count INTEGER NOT NULL,
    seeds TEXT NOT NULL,                   -- JSON array of floats
    mean REAL NOT NULL,
    std REAL NOT NULL,
    baseline_id TEXT,                      -- FK to experiments.id
    baseline_mean REAL,
    baseline_seed_count INTEGER,
    p_value REAL,
    effect_size REAL,
    validation_status TEXT NOT NULL,      -- ValidationStatus enum
    leakage_status TEXT,                   -- LeakageStatus enum
    claimed_improvement REAL,
    claimed_confidence TEXT,
    claim_source TEXT,
    claim_verified INTEGER DEFAULT 0,
    conditions TEXT NOT NULL,              -- JSON
    metadata TEXT,                         -- JSON
    superseded_by TEXT,                    -- FK to experiments.id
    invalidation_reason TEXT,
    created_at TEXT NOT NULL,              -- ISO timestamp
    updated_at TEXT NOT NULL,
    FOREIGN KEY (baseline_id) REFERENCES experiments(id),
    FOREIGN KEY (superseded_by) REFERENCES experiments(id)
);

-- Indexes for query performance
CREATE INDEX idx_exp_technique ON experiments(technique_id);
CREATE INDEX idx_exp_status ON experiments(validation_status);
CREATE INDEX idx_exp_leakage ON experiments(leakage_status);
CREATE INDEX idx_exp_outcome_type ON experiments(outcome_type);
CREATE INDEX idx_exp_created ON experiments(created_at);

-- Claims table
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    experiment_result_id TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    claimed_delta REAL NOT NULL,
    claimed_confidence TEXT,
    evidence_status TEXT NOT NULL,
    multi_seed_required INTEGER NOT NULL,
    minimum_seeds_for_validation INTEGER NOT NULL,
    FOREIGN KEY (experiment_result_id) REFERENCES experiments(id)
);

CREATE INDEX idx_claim_experiment ON claims(experiment_result_id);
CREATE INDEX idx_claim_status ON claims(evidence_status);

-- Validation events (audit log)
CREATE TABLE validation_events (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT,
    actor TEXT NOT NULL,
    evidence TEXT,                         -- JSON
    timestamp TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX idx_ve_experiment ON validation_events(experiment_id);
CREATE INDEX idx_ve_timestamp ON validation_events(timestamp);

-- Reproduction records
CREATE TABLE reproductions (
    id TEXT PRIMARY KEY,
    technique_id TEXT NOT NULL,
    original_experiment_id TEXT NOT NULL,
    reproducer TEXT NOT NULL,
    result_bpb REAL,
    result_match INTEGER NOT NULL,
    delta_from_original REAL,
    hardware TEXT,
    software_stack TEXT,
    notes TEXT,
    conditions TEXT NOT NULL,              -- JSON
    confidence TEXT NOT NULL,              -- HIGH, MEDIUM, LOW
    timestamp TEXT NOT NULL,
    FOREIGN KEY (original_experiment_id) REFERENCES experiments(id)
);

CREATE INDEX idx_rep_technique ON reproductions(technique_id);
CREATE INDEX idx_rep_original ON reproductions(original_experiment_id);

-- Leakage reports
CREATE TABLE leakage_reports (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    clean INTEGER NOT NULL,
    overall_category TEXT NOT NULL,
    impact_validity TEXT NOT NULL,
    report_json TEXT NOT NULL,             -- Full violation details as JSON
    timestamp TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX idx_lr_experiment ON leakage_reports(experiment_id);
```

**Why SQLite**:
- Indexed queries for "all techniques with SINGLE_SEED_PENDING" — O(log n) vs O(n) for JSONL scan
- Deduplication via UNIQUE constraints on (technique_id, conditions_hash)
- Schema migrations via ALTER TABLE for new fields
- ACID transactions for data integrity
- Single file, portable, git-friendly at small scale (scales to ~100K experiments before needing migration to Postgres)

**Deduplication Strategy**:
```python
def conditions_hash(conditions: ExperimentConditions) -> str:
    """Stable hash of canonicalized conditions for deduplication."""
    canonical = {
        "pr_number": conditions.pr_number,
        "ttt_optimizer": conditions.ttt_optimizer,
        "quantization": conditions.quantization,
        # ... other relevant fields
    }
    return hashlib.sha1(json.dumps(canonical, sort_keys=True)).hexdigest()[:12]
```

If an experiment with the same technique_id and conditions_hash already exists:
- If the new result has MORE seeds, update the existing record
- If the new result has FEWER seeds, reject as duplicate
- This prevents double-counting while allowing seed accumulation

### A2. Validation Rules

#### Minimum Seed Count Thresholds

| Claim Type | Minimum Seeds | Rationale |
|------------|---------------|-----------|
| Routine ablation (confirming established technique) | 2 | Detect gross errors |
| Improvement claim over baseline | 3 | p-value meaningful |
| Record-breaking / novel technique | 5 | Strong evidence bar |
| Competition submission | 5+ | Significance at p < 0.01 |

**Hard rule**: Claims in the knowledge base without seed_count >= 3 are labeled `SINGLE_SEED_PENDING` and cannot be marked `VALIDATED_SIGNIFICANT`.

#### Statistical Tests (FIXED: Reconciled Thresholds — Addresses Critique A1)

**Primary**: Welch's t-test (unequal variances assumed)

```python
from scipy import stats
import numpy as np

def welch_ttest(experiment_seeds, baseline_seeds, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(experiment_seeds, baseline_seeds,
                                        equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value, "significant": p_value < alpha}

def cohens_d(experiment_seeds, baseline_seeds):
    """Cohen's d for independent samples with unequal variances."""
    n1, n2 = len(experiment_seeds), len(baseline_seeds)
    var1, var2 = np.var(experiment_seeds, ddof=1), np.var(baseline_seeds, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(experiment_seeds) - np.mean(baseline_seeds)) / pooled_std

def pooled_std_approx(seeds1: list[float], seeds2: list[float]) -> float:
    """Conservative pooled std for small samples."""
    n1, n2 = len(seeds1), len(seeds2)
    var1 = np.var(seeds1, ddof=1) if n1 > 1 else 0
    var2 = np.var(seeds2, ddof=1) if n2 > 1 else 0
    if n1 <= 1 and n2 <= 1:
        # Both single seeds — use typical BPB noise as conservative estimate
        return 0.003  # ~0.003 BPB typical std for parameter golf
    elif n1 <= 1:
        return np.sqrt(var2)  # Use single known std
    elif n2 <= 1:
        return np.sqrt(var1)
    else:
        return np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
```

**Reconciled Significance Criteria (FIXED)**:

The original plan had conflicting requirements:
- Line 173: `Effect size |d| > 0.3 OR absolute delta > 0.001 BPB`
- Lines 176-179: `delta < -0.001 BPB with p < 0.05`

**Resolution**: These are NOT simultaneously required. They serve different purposes:

| Criterion | Purpose | Requirement |
|-----------|---------|-------------|
| p < 0.05 | Statistical significance | Must be met for VALIDATED_SIGNIFICANT |
| \|d\| > 0.3 | Practical significance | OR with delta threshold — ensures effect is not trivially small |
| \|delta\| > 0.001 BPB | BPB practical threshold | OR with effect size — ensures minimum meaningful magnitude |

**For VALIDATED_SIGNIFICANT, ALL of the following must hold**:

1. **Statistical**: p-value < 0.05 (Welch's t-test, two-sided)
2. **Practical magnitude** (ONE of):
   - Effect size |d| > 0.3 (small effect minimum), OR
   - |delta| > 0.001 BPB for BPB metrics (accounts for typical noise floor)
3. **Reproducibility**: std < |delta| (means separated by > 1 std), OR seed_count >= 5 with p < 0.01

**BPB-Specific Thresholds** (context-dependent, not contradictory):

| Context | Delta Threshold | P-value | Effect Size | Additional |
|---------|-----------------|---------|-------------|------------|
| Routine adoption | < -0.0005 BPB | < 0.05 | > 0.3 | — |
| Improvement claim | < -0.001 BPB | < 0.05 | > 0.3 OR \|d\| > 0.2 | std < \|delta\| |
| Competition submission | < -0.005 BPB | < 0.01 | > 0.5 | 5+ seeds |
| Record-breaking | < -0.010 BPB | < 0.01 | > 0.8 | 5+ seeds |

**The key insight**: Effect size (d) and absolute delta (0.001 BPB) are alternatives for demonstrating practical significance, not joint requirements. A tiny effect (|d| = 0.15) could still be meaningful if the delta is large in absolute terms (e.g., -0.010 BPB), and vice versa.

**Mathematical reconciliation**:

Given typical BPB noise (std = 0.002-0.005):
- |d| > 0.3 requires |delta| > 0.3 * std ≈ 0.0006-0.0015 BPB
- The 0.001 BPB threshold is at the boundary of detectability given typical noise
- For a std of 0.003 BPB: |d| > 0.3 means |delta| > 0.0009 BPB
- This is consistent with the 0.001 BPB practical threshold

The original plan's problem was stating them as joint requirements ("ALL must be met") when they should be alternatives within the "practical significance" prong.

#### Mixed Signals Detection

```python
def detect_mixed_signals(seeds: list[float], baseline_mean: float) -> bool:
    """
    Returns True if seeds show inconsistent signs relative to baseline.
    e.g., [1.100, 1.102, 1.098] vs baseline 1.105 → True (2/3 positive)
    """
    n_better = sum(1 for s in seeds if s < baseline_mean)
    n_worse = sum(1 for s in seeds if s > baseline_mean)
    n_same = sum(1 for s in seeds if s == baseline_mean)

    # Mixed if both positive and negative signals exist
    return n_better > 0 and n_worse > 0

def compute_signal_ratio(seeds: list[float], baseline_mean: float) -> tuple[int, int]:
    """Returns (n_better, n_worse) counts."""
    n_better = sum(1 for s in seeds if s < baseline_mean)
    n_worse = sum(1 for s in seeds if s > baseline_mean)
    return (n_better, n_worse)
```

#### Variance Thresholds

| Variance (std) | Interpretation | Action |
|----------------|----------------|--------|
| std < 0.001 BPB | Very stable | Accept as validated |
| 0.001 <= std < 0.005 BPB | Normal | Standard validation |
| 0.005 <= std < 0.01 BPB | High variance | Flag, require more seeds |
| std >= 0.01 BPB | Very high | Mark INCONCLUSIVE, investigate |

### A3. Baseline Requirements (FIXED: Addresses Critique A3)

#### Canonical Baseline Definition

A **canonical baseline** is the best-validated existing result for a given technique family, determined by:

1. **Validation status hierarchy** (highest to lowest):
   - VALIDATED_SIGNIFICANT with most seeds
   - VALIDATED_SIGNIFICANT
   - MULTI_SEED_PENDING with most seeds
   - SINGLE_SEED_PENDING

2. **When comparing two techniques**, the baseline is:
   - The most validated predecessor technique in the dependency graph, OR
   - The most recent VALIDATED_SIGNIFICANT result in the same technique family

#### Baseline Chaining Rules

**Rule 1: SINGLE_SEED_PENDING baselines cannot validate multi-seed experiments**
```python
def can_baseline_validate(baseline: ExperimentResult, experiment: ExperimentResult) -> bool:
    if baseline.validation_status == ValidationStatus.SINGLE_SEED_PENDING:
        if experiment.seed_count > 1:
            return False  # Single-seed baseline cannot validate multi-seed claim
    return True
```

**Rule 2: Multi-seed baseline preferred for multi-seed experiments**
```python
def select_baseline_candidates(technique_id: UUID) -> list[ExperimentResult]:
    """Return baselines sorted by suitability for comparison."""
    candidates = find_experiments_by_technique(technique_id)
    # Sort by: validation_status (VALIDATED_SIGNIFICANT first), seed_count (desc), recency
    return sorted(candidates, key=lambda e: (
        e.validation_status != ValidationStatus.VALIDATED_SIGNIFICANT,
        -e.seed_count,
        -e.created_at
    ))
```

**Rule 3: Cross-experiment validation requires compatible conditions**
```python
def conditions_compatible(exp1: ExperimentResult, exp2: ExperimentResult) -> bool:
    """Check if two experiments have comparable conditions."""
    c1, c2 = exp1.conditions, exp2.conditions
    # Must match on key variables; allow variation on minor ones
    key_fields = ['quantization', 'model_size', 'ttt_optimizer', 'data_split']
    return all(getattr(c1, f) == getattr(c2, f) for f in key_fields)
```

**Rule 4: Baseline can be upgraded as evidence accumulates**
If a SINGLE_SEED_PENDING experiment later gets multiple seeds that validate it, that experiment becomes a valid baseline for subsequent comparisons.

### A4. Retrospective Invalidation Protocol (NEW: Addresses Critique Combined C1)

When new evidence contradicts an earlier assessment, the following protocol applies:

```python
class InvalidationEvent(BaseModel):
    id: UUID
    original_experiment_id: UUID
    superseding_experiment_id: UUID | None
    invalidation_type: InvalidationType
    reason: str
    timestamp: datetime
    actor: str                              # "system", "user", "community"
    confidence: str                         # HIGH, MEDIUM, LOW

class InvalidationType(str, Enum):
    REVERSED_BY_SEEDS      = "REVERSED_BY_SEEDS"       # New seeds contradict old result
    LEAKAGE_DISCOVERED     = "LEAKAGE_DISCOVERED"      # Category 1/2 violation found post-hoc
    BETTER_BASELINE        = "BETTER_BASELINE"         # New validated baseline changes comparison
    RETRACTED_BY_AUTHOR    = "RETRACTED_BY_AUTHOR"
    CONDITIONS_CHANGED     = "CONDITIONS_CHANGED"      # Found conditions not actually equivalent

async def invalidate_experiment(
    experiment_id: UUID,
    invalidation: InvalidationEvent
) -> ExperimentResult:
    """
    Invalidation protocol:
    1. Mark original experiment as RETRACTED
    2. Record superseding experiment if applicable
    3. Propagate invalidation to dependent techniques
    4. Emit audit event
    5. Notify via QKayV warning if the technique is in active use
    """
    original = db.get_experiment(experiment_id)

    # Mark as retracted
    original.validation_status = ValidationStatus.RETRACTED
    original.superseded_by = invalidation.superseding_experiment_id
    original.invalidation_reason = invalidation.reason

    # Propagate to dependent techniques
    dependents = dependency_graph.get_dependents(original.technique_id)
    for dep in dependents:
        # Re-evaluate dependent's confidence
        await reevaluate_validation_status(dep.id)

    # Emit warning for active stacks
    active_stacks = find_stacks_using_technique(original.technique_id)
    for stack in active_stacks:
        emit_warning(f"Technique {original.technique_id} in stack {stack.id} has been invalidated: {invalidation.reason}")

    return original
```

**Invalidation Propagation Rules**:
- A technique's invalidation does NOT automatically invalidate its dependents
- Dependents must be re-evaluated against the new information
- If dependent's confidence was partially based on invalidated technique, reduce confidence accordingly

### A5. Query Interface

#### Q1: "Is this finding statistically validated?"

```python
def query_validation_status(technique_id: str) -> ValidationStatusResponse:
    result = find_experiment(technique_id)
    if not result:
        return {"found": False}

    signal_ratio = None
    if result.seed_count >= 3:
        n_better, n_worse = compute_signal_ratio(result.seeds, result.baseline_mean)
        signal_ratio = f"{n_better}/{result.seed_count}"

    return {
        "technique": result.technique,
        "validation_status": result.validation_status,
        "seed_count": result.seed_count,
        "signal_ratio": signal_ratio,        # NEW: "2/3" for MIXED_SIGNALS
        "mean": result.mean,
        "std": result.std,
        "p_value": result.p_value,
        "effect_size": result.effect_size,
        "ci_95": result.confidence_interval,
        "validation_evidence": get_validation_evidence(result.id),
        "reproductions": count_reproductions(technique_id),
        "leakage_status": result.leakage_status,
    }
```

**Example response (MIXED_SIGNALS)**:
```
Technique: Experimental TTT Variant X
Validation Status: MIXED_SIGNALS

Seed count: 3
Signal ratio: 2/3 seeds support improvement (1/3 shows regression)

Results:
- Seed 1: 1.102 BPB (better than baseline 1.105)
- Seed 2: 1.103 BPB (better than baseline)
- Seed 3: 1.108 BPB (WORSE than baseline)

Mean: 1.104 BPB, std: 0.003 BPB
p-value: 0.34 (not significant)

Recommendation: The claimed improvement is NOT validated.
2 of 3 seeds show improvement, but the effect is inconsistent.
Possible causes: hardware variance, data split sensitivity, or the
technique is sensitive to initialization. Investigate seed-level
conditions before treating this as a genuine improvement.
```

#### Q2: "Should I run more seeds before acting on this result?"

```python
def query_seed_recommendation(technique_id: str) -> SeedRecommendation:
    result = find_experiment(technique_id)

    if result.validation_status == ValidationStatus.MIXED_SIGNALS:
        return {
            "recommendation": "INVESTIGATE_INSTEAD",
            "priority": "HIGH",
            "reason": "Mixed signals across seeds — running more seeds without understanding the variance source is inefficient",
            "suggested_action": "Audit seed-level conditions: hardware, data splits, initialization",
        }

    # Standard sigma_gap logic
    delta = abs(result.claimed_improvement or 0)
    std = result.std

    if std == 0:
        return {
            "recommendation": "RUN_MORE_SEEDS",
            "priority": "HIGH",
            "reason": "Single seed; cannot estimate variance",
            "suggested_seeds": 5
        }

    sigma_gap = delta / std if std > 0 else float('inf')

    if sigma_gap > 3 and result.seed_count >= 3:
        return {
            "recommendation": "ADEQUATE_SEEDS",
            "priority": "LOW",
            "confidence": "HIGH"
        }
    elif result.seed_count >= 3:
        return {
            "recommendation": "RUN_MORE_SEEDS",
            "priority": "MEDIUM",
            "suggested_seeds": max(5, result.seed_count + 2)
        }
    else:
        return {
            "recommendation": "RUN_MORE_SEEDS",
            "priority": "HIGH",
            "suggested_seeds": 5
        }
```

---

## Part B: Information Leakage Detector

### B1. Taxonomy of Leakage (from Issue #402)

#### Category 1: Invalid (Direct Data Leakage)

**Definition**: Training or adaptation directly on validation/test data.

**Examples**:
- Running gradient descent on validation set tokens
- Updating model weights using validation loss
- Using validation data in any learned adaptation step

**Severity**: CRITICAL — results are fundamentally compromised

#### Category 2: Gray Zone (Token-Stream TTT without Isolation)

**Definition**: Score-first TTT where the token stream is not properly document-isolated.

**Scale-Dependent Severity** (NEW):
The severity of Category 2 depends on the number of documents processed:

| Document Count | Severity | Rationale |
|----------------|----------|-----------|
| < 10 | LOW | Information bleed is negligible |
| 10-100 | MEDIUM | Accumulating concern |
| > 100 | HIGH | Substantial cross-document leakage |
| > 1000 | CRITICAL | Treat as Category 1 equivalent |

**Why scale matters**: At small scale, the probability that any specific token's information persists into the model's adaptation is low. At large scale (100K+ tokens), even small per-token leakage accumulates.

**The Problem at Scale**:
```
Token 1-1000: Document A
Token 1001-2000: Document B
...
Token at 1001 adapts using information that includes Document A's tokens
This creates implicit leakage: the model adapts to Document B using
information from Document A's validation tokens
```

#### Category 3: Valid (Document-Independent Score-First TTT)

**Definition**: Each document is scored and adapted in complete isolation, under `torch.inference_mode()`, before any weight updates occur.

**Key Invariant**: Every token must be scored under `torch.inference_mode()` BEFORE any weight update.

### B2. Pre-Flight Advisory System (FIXED: Addresses Critique B1)

The original "prevention" framing was misleading. QKayV cannot prevent users from writing non-compliant training code. Instead, we provide **advisory pre-flight checks** that:

1. Validate structured configuration files against a schema
2. Check for known dangerous patterns in code snippets the user provides
3. Cannot guarantee runtime compliance but reduce errors

```python
class LeakagePreFlightAdvisory:
    """
    Pre-experiment advisory checks for TTT compliance.
    NOT a prevention system — users can override warnings.
    """

    def __init__(self):
        self.schema_validator = ConfigSchemaValidator()
        self.pattern_matcher = CodePatternMatcher()

    def check_config(self, config: dict) -> AdvisoryResult:
        """
        Validate a structured config file for common TTT compliance issues.
        This CAN catch many configuration errors before runtime.
        """
        issues = []

        # Check 1: TTT phase ordering in config
        if config.get('ttt_enabled'):
            phase_order = config.get('ttt_phase_order', [])
            expected = ['score', 'adapt']
            if phase_order != expected:
                issues.append(AdvisoryIssue(
                    severity="HIGH",
                    category="CATEGORY_1",
                    check="ttt_phase_order",
                    message=f"TTT phase order is {phase_order}, expected {expected}",
                    fix="Set ttt_phase_order to ['score', 'adapt']"
                ))

        # Check 2: inference_mode flag during scoring
        if not config.get('inference_mode_during_scoring', True):
            issues.append(AdvisoryIssue(
                severity="HIGH",
                category="CATEGORY_2_OR_3",
                check="inference_mode_during_scoring",
                message="inference_mode_during_scoring is False — scoring may leak information"
            ))

        # Check 3: Document isolation setting
        if not config.get('isolate_documents', False):
            issues.append(AdvisoryIssue(
                severity="MEDIUM",
                category="CATEGORY_2",
                check="document_isolation",
                message="Documents are not isolated — consider enabling for Category 3 compliance"
            ))

        return AdvisoryResult(
            cleared=len([i for i in issues if i.severity == "HIGH"]) == 0,
            issues=issues,
            compute_risk=config.get('estimated_compute_cost', 0)
        )

    def check_code_snippet(self, code: str) -> AdvisoryResult:
        """
        Analyze a Python code snippet for known dangerous patterns.
        Uses pattern matching — cannot guarantee completeness.
        """
        issues = []

        dangerous_patterns = [
            (r'model\.train\(\).*?for.*?in.*?val', "CATEGORY_1",
             "Training loop iterates over validation data — likely leakage"),
            (r'loss\.backward\(\).*?val', "CATEGORY_1",
             "Gradient computed on validation — this is direct leakage"),
            (r'torch\.inference_mode\(\).*?score.*?adapt.*?score',
             "CATEGORY_2", "Adaptation between scoring phases — verify document isolation"),
        ]

        for pattern, category, message in dangerous_patterns:
            matches = re.finditer(pattern, code, re.DOTALL)
            for match in matches:
                issues.append(AdvisoryIssue(
                    severity="HIGH" if category == "CATEGORY_1" else "MEDIUM",
                    category=category,
                    check="code_pattern",
                    message=message,
                    code_snippet=match.group(0)[:200]  # First 200 chars
                ))

        return AdvisoryResult(
            cleared=len([i for i in issues if i.severity == "HIGH"]) == 0,
            issues=issues,
            compute_risk=0  # Pre-flight has no compute cost
        )

class AdvisoryResult(BaseModel):
    cleared: bool                          # True if no HIGH severity issues
    issues: list[AdvisoryIssue]
    compute_risk: float                    # Estimated compute wasted if issues ignored

class AdvisoryIssue(BaseModel):
    severity: str                          # HIGH, MEDIUM, LOW
    category: str                          # CATEGORY_1, CATEGORY_2, CATEGORY_3
    check: str                             # Which check found this
    message: str
    fix: str | None
    code_snippet: str | None
```

**Scope Limitations**:
- QKayV CAN validate structured config files against a schema
- QKayV CAN pattern-match against known dangerous code patterns
- QKayV CANNOT statically analyze arbitrary Python training loops with certainty
- Runtime behavior can only be verified through log analysis (see B3)

### B3. Log Parser Specification (NEW: Addresses Critique B4)

Phase 2's key deliverable is a log parser. This section specifies the input formats.

#### Supported Log Formats

**Format 1: Structured JSON Logs (Preferred)**

```python
class StructuredStepLog(BaseModel):
    """Standard format for QKayV-compatible training logs."""
    step_number: int
    phase: str                             # "initial_eval", "ttt_scoring", "ttt_adaptation", "final_eval"
    timestamp: str                        # ISO 8601

    # Phase-specific fields
    document_id: str | None               # For TTT phases
    document_start_token: int | None
    document_end_token: int | None

    # Inference mode tracking
    inference_mode_active: bool
    no_grad_active: bool

    # Gradient tracking
    gradients_computed: bool
    gradient_norm: float | None
    updated_params: list[str] | None      # Parameter names updated

    # Scores (for scoring phases)
    per_token_scores: list[float] | None
    mean_score: float | None

    # Model state
    weights_modified: bool
    modified_param_names: list[str] | None

class StructuredExperimentLog(BaseModel):
    experiment_id: str
    pr_number: int | None
    technique: str
    model_config: dict
    conditions: ExperimentConditions
    steps: list[StructuredStepLog]
    metadata: dict
```

**Format 2: PyTorch Training Logger Output (Extracted)**

For existing training scripts that don't emit structured logs, QKayV provides a logger utility:

```python
# In user's training script
from qkv.training_logger import QKayVLogger

logger = QKayVLogger(experiment_id="pr_528", technique="Full GPTQ + TTT")

for step in training_loop:
    logger.log_phase(
        phase="ttt_scoring",
        document_id=f"doc_{i}",
        inference_mode_active=torch.is_inference_mode_enabled(),
        gradients_computed=model.has_gradients(),
        scores=per_token_scores,
    )
    logger.log_phase(
        phase="ttt_adaptation",
        updated_params=list(model.updated_param_names()),
    )
```

**Format 3: Legacy Log Conversion**

For PRs with unstructured logs, a conversion tool extracts structured fields:

```python
class LegacyLogConverter:
    """
    Converts unstructured training logs to StructuredExperimentLog format.
    Uses heuristics and pattern matching — not guaranteed complete.
    """

    def convert(self, log_text: str) -> StructuredExperimentLog:
        # Parse timestamps
        steps = self._extract_steps(log_text)

        # Infer phases from log patterns
        for step in steps:
            step.phase = self._infer_phase(step)

        # Detect inference_mode from log context
        for step in steps:
            step.inference_mode_active = self._detect_inference_mode(step)

        return StructuredExperimentLog(steps=steps)

    def _infer_phase(self, step: ParsedStep) -> str:
        """Infer phase from log patterns."""
        text = step.raw_text.lower()
        if 'score' in text or 'eval' in text:
            return "ttt_scoring"
        elif 'adapt' in text or 'train' in text:
            return "ttt_adaptation"
        elif 'initial' in text:
            return "initial_eval"
        else:
            return "unknown"
```

**Parser Output Guarantee**:
- If structured logger was used: 100% field fidelity
- If legacy conversion was used: fields are best-effort; unknown fields marked as `None`
- Confidence score attached to each converted log

### B4. Detection Patterns

Same as original plan's Patterns 1-4, operating on `StructuredExperimentLog` format.

#### Scale-Dependent Category 2 Assessment

```python
def assess_category2_severity(
    log: StructuredExperimentLog,
    document_count: int
) -> tuple[str, str]:
    """
    Assess Category 2 severity based on document count and isolation.
    Returns (severity, rationale).
    """
    isolated = all(
        step.document_id is not None
        for step in log.steps
        if step.phase == "ttt_scoring"
    )

    if not isolated:
        if document_count > 1000:
            return ("CRITICAL", "Token-stream TTT with >1000 documents — treat as Category 1")
        elif document_count > 100:
            return ("HIGH", "Token-stream TTT with >100 documents — substantial leakage risk")
        elif document_count > 10:
            return ("MEDIUM", "Token-stream TTT with moderate document count")
        else:
            return ("LOW", "Token-stream TTT with few documents — leakage negligible")

    return ("NONE", "Documents properly isolated — no Category 2 concern")
```

### B5. Leakage Status Schema

```python
class LeakageStatus(str, Enum):
    CLEAN                = "CLEAN"                # No violations detected
    CATEGORY_1           = "CATEGORY_1"           # Direct train on validation
    CATEGORY_2_DETECTED  = "CATEGORY_2_DETECTED"  # Token-stream without isolation
    CATEGORY_2_SCALE_RISK = "CATEGORY_2_SCALE_RISK" # Category 2 at scale
    CATEGORY_3_BORDERLINE = "CATEGORY_3_BORDERLINE" # Minor concerns, mostly valid

class LeakageReport(BaseModel):
    id: UUID
    experiment_id: UUID
    clean: bool
    overall_category: LeakageStatus
    impact_validity: str                      # VALID, INVALID, DEGRADED, VALID_WITH_CONCERNS
    document_count: int | None
    category2_severity: str | None           # Scale-dependent severity
    violations: list[LeakageViolation]
    confidence: str                           # HIGH, MEDIUM, LOW
    timestamp: datetime

class LeakageViolation(BaseModel):
    type: str                                 # CATEGORY_1, CATEGORY_2, CATEGORY_3_BORDERLINE
    severity: str                             # CRITICAL, HIGH, MEDIUM, LOW
    step: int | None
    description: str
    details: str | None
    fix: str | None
```

---

## Part C: Combined Implementation

### C1. Composite Status (NEW)

The combination of validation status (Part A) and leakage status (Part B) produces a composite confidence:

```python
def composite_confidence(
    validation_status: ValidationStatus,
    leakage_status: LeakageStatus
) -> CompositeConfidence:
    """
    Combine validation and leakage status into a single confidence score.
    """

    # If leakage is Category 1, nothing else matters
    if leakage_status == LeakageStatus.CATEGORY_1:
        return CompositeConfidence(
            level="INVALID",
            score=0.0,
            reason="CATEGORY_1_LEAKAGE: Results are fundamentally compromised"
        )

    # Map validation status to score
    validation_scores = {
        ValidationStatus.VALIDATED_SIGNIFICANT: 1.0,
        ValidationStatus.MULTI_SEED_PENDING: 0.8,
        ValidationStatus.MIXED_SIGNALS: 0.5,
        ValidationStatus.SINGLE_SEED_PENDING: 0.3,
        ValidationStatus.INCONCLUSIVE: 0.2,
        ValidationStatus.VALIDATED_NULL: 0.0,
        ValidationStatus.VALIDATED_NEGATIVE: -0.5,
        ValidationStatus.RETRACTED: 0.0,
    }

    base_score = validation_scores.get(validation_status, 0.0)

    # Leakage adjustments
    leakage_multipliers = {
        LeakageStatus.CLEAN: 1.0,
        LeakageStatus.CATEGORY_3_BORDERLINE: 0.9,
        LeakageStatus.CATEGORY_2_SCALE_RISK: 0.7,
        LeakageStatus.CATEGORY_2_DETECTED: 0.5,
    }

    multiplier = leakage_multipliers.get(leakage_status, 1.0)
    final_score = base_score * multiplier

    # Determine level
    if final_score >= 0.8:
        level = "HIGH"
    elif final_score >= 0.5:
        level = "MEDIUM"
    elif final_score >= 0.2:
        level = "LOW"
    else:
        level = "UNUSABLE"

    return CompositeConfidence(
        level=level,
        score=final_score,
        validation_component=base_score,
        leakage_multiplier=multiplier,
        reason=f"Validation {validation_status.value} * Leakage {leakage_status.value}"
    )

class CompositeConfidence(BaseModel):
    level: str                               # HIGH, MEDIUM, LOW, UNUSABLE
    score: float                             # 0.0 to 1.0
    validation_component: float
    leakage_multiplier: float
    reason: str
```

### C2. Implementation Roadmap (FIXED: Addresses Critique C2)

#### Phase 1: Metadata Schema + SQLite Storage + Manual Tracking (Weeks 1-3)

**Goal**: Establish the data model and allow manual experiment entry.

**Deliverables**:
1. SQLite database schema with all indexes
2. `ExperimentResult`, `ClaimRecord`, `ReproductionRecord`, `ValidationEvent` schemas
3. CLI commands for manual experiment entry:
   - `qkv exp add --technique X --outcome 1.123 --seed-count 3 --seeds 1.123,1.124,1.122`
   - `qkv exp add --technique X --claim "improves by 0.01 BPB" --pr 490`
4. Basic query interface:
   - `qkv exp status X` — returns validation status
   - `qkv exp validate X --baseline Y` — runs statistical tests
5. Deduplication logic for duplicate experiment detection

**Milestone**: User can manually enter experiments and query their validation status. Storage is indexed and scalable.

**Estimation basis**: Schema design and SQLite setup is well-understood. 3 weeks allows for testing edge cases.

#### Phase 2: Structured Logging + Log Parser + Leakage Detection (Weeks 4-9)

**Goal**: Parse experiment logs automatically and detect leakage patterns.

**Deliverables**:
1. Structured logging library (`qkv.training_logger`) for use in training scripts
2. Legacy log converter for unstructured logs
3. Leakage detector operating on StructuredExperimentLog:
   - Pattern matchers for CATEGORY_1, CATEGORY_2, CATEGORY_3
   - Scale-dependent Category 2 assessment
   - TTT phase ordering validator
4. Pre-flight advisory tool:
   - Config file validator
   - Code snippet pattern matcher
5. PR metadata extractor:
   - Parse PR descriptions for claimed BPB values
   - Extract seed counts from experiment configurations
6. Query interface expansion:
   - `qkv leakage scan --pr 532` — assess PR for leakage
   - `qkv leakage check-config --file config.yaml` — pre-flight advisory

**Why 6 weeks (not 2)**: Parsing diverse training scripts to extract structured `phase`, `gradients_computed`, `document_boundaries_respected` fields is a significant program analysis task. Each parameter golf PR has idiosyncratic code structure. The legacy log converter requires iterative development with real PRs as test cases.

**Milestone**: QKayV can parse a structured training log or legacy log and report leakage status with confidence rating.

#### Phase 3: Proactive Validation Planning + Integrated Recommendations (Weeks 10-13)

**Goal**: Actively suggest follow-up experiments and integrate validation/leakage into recommendations.

**Deliverables**:
1. Seed recommendation engine:
   - Given a technique with N seeds, recommend additional seeds
   - Based on effect size, variance, and desired confidence
2. Follow-up experiment planner:
   - `qkv exp plan-validation --technique X` — generates validation experiment spec
   - Estimates compute cost
   - Specifies exact seeds to run
3. Composite confidence scoring integration:
   - Integrate validation status into technique scoring
   - Integrate leakage status into recommendation eligibility
4. Retrospective invalidation automation:
   - Detect when new evidence contradicts earlier assessments
   - Emit warnings for techniques in active stacks
5. Knowledge base integration:
   - Validated techniques gain higher recommendation weight
   - Leaky techniques are flagged or excluded
   - Failure mode KB enriched with validation findings
6. Reporting:
   - `qkv report validation-gaps` — lists all unvalidated techniques in KB
   - `qkv report leakage-summary` — summarizes leakage findings across all PRs

**Milestone**: QKayV proactively guides users toward validated techniques and flags unvalidated ones. Invalidated techniques trigger warnings in active stacks.

#### Revised Timeline Summary

| Phase | Duration | Core Capability | Key Deliverables |
|-------|----------|-----------------|------------------|
| 1 | Weeks 1-3 | Metadata schema + SQLite + manual tracking | SQLite store, CLI entry, basic queries, deduplication |
| 2 | Weeks 4-9 | Structured logging + log parser + leakage detection | Logger lib, legacy converter, leakage detector, pre-flight advisory |
| 3 | Weeks 10-13 | Proactive validation + integrated recommendations | Seed planner, composite confidence, invalidation protocol |

**Total: 13 weeks (vs original 6 weeks)**

The original Phase 2 estimate of 2 weeks was unrealistic for the log parser task. The 6-week estimate for Phase 2 reflects:
- 2 weeks: Structured logging library + adoption
- 2 weeks: Legacy log converter with iterative testing
- 2 weeks: Leakage detector refinement + pre-flight advisory

### C3. Integration with Existing QKayV

**Code location**: `qkv/validation/` module

```
qkv/
  validation/
    __init__.py
    schemas.py          # ExperimentResult, ClaimRecord, etc.
    database.py        # SQLite interface
    statistics.py      # Welch's t-test, Cohen's d, bootstrap CI
    validation.py      # ValidationStatus logic
    leakage/
      __init__.py
      detector.py      # Pattern matching logic
      preflight.py     # Pre-flight advisory
      parser.py        # Log parser
      scales.py        # Scale-dependent severity
    queries.py         # Query interface functions
    cli.py              # CLI commands
```

**Integration points**:
- Dependency graph: technique validation status propagates to dependents
- Failure mode KB: leakage violations become failure mode entries
- Recommendation engine: composite confidence score influences technique scoring

---

## Appendix: Reference Tables

### A. Validation Status Transitions (Updated)

```
SINGLE_SEED_PENDING
  ├── Run 2+ seeds, all similar → MULTI_SEED_PENDING
  ├── Run 2+ seeds, mixed signals → MIXED_SIGNALS
  ├── Run 2+ seeds, high variance → INCONCLUSIVE
  └── Run 1 seed, widely reproduced → VALIDATED_SIGNIFICANT (via reproduction)

MULTI_SEED_PENDING
  ├── p < 0.05, effect size adequate → VALIDATED_SIGNIFICANT
  ├── p >= 0.05 → VALIDATED_NULL
  ├── Mixed signals → MIXED_SIGNALS
  └── High variance, underpowered → INCONCLUSIVE

MIXED_SIGNALS
  ├── Investigation explains heterogeneity → VALIDATED_SIGNIFICANT or VALIDATED_NULL
  └── Remains unexplained → MIXED_SIGNALS (flag for manual review)

VALIDATED_SIGNIFICANT
  └── Later contradicted by better study → RETRACTED → re-evaluate

INCONCLUSIVE
  ├── More seeds resolve → VALIDATED_SIGNIFICANT or VALIDATED_NULL
  └── More seeds keep inconclusive → INCONCLUSIVE (and flag as hard case)

RETRACTED
  └── Superseded by new validated result → Depends on new evidence
```

### B. Leakage Category Summary

| Category | Name | Severity | Description |
|----------|------|----------|-------------|
| 1 | Invalid | CRITICAL | Train/direct adapt on validation data |
| 2 | Gray | MEDIUM-HIGH (scale-dependent) | Token-stream TTT without document isolation |
| 3 | Valid | — | Document-independent score-first TTT |

### C. TTT Validity Checklist

For any TTT implementation, verify ALL of:

- [ ] `torch.inference_mode()` OR `torch.no_grad()` is active during ALL scoring
- [ ] Every token is scored BEFORE any weight update in the TTT process
- [ ] Documents are processed in isolation (not merged token streams) — especially important at scale (>100 docs)
- [ ] Scoring phase and adaptation phase are strictly ordered (score → adapt, never adapt → score)
- [ ] Embeddings are frozen OR verified safe from catastrophic forgetting
- [ ] No gradient computation on validation data

### D. Composite Confidence Levels

| Validation Status | Leakage Status | Composite Level | Score |
|-------------------|----------------|-----------------|-------|
| VALIDATED_SIGNIFICANT | CLEAN | HIGH | 1.0 |
| VALIDATED_SIGNIFICANT | CATEGORY_3_BORDERLINE | HIGH | 0.9 |
| MULTI_SEED_PENDING | CLEAN | MEDIUM | 0.8 |
| VALIDATED_SIGNIFICANT | CATEGORY_2_DETECTED | MEDIUM | 0.5 |
| MIXED_SIGNALS | CLEAN | LOW | 0.5 |
| SINGLE_SEED_PENDING | CLEAN | LOW | 0.3 |
| Any | CATEGORY_1 | UNUSABLE | 0.0 |

### E. Parameter Golf PR Validation Status Reference (Updated)

| PR | BPB | Technique | Validation Status | Composite Confidence | Notes |
|----|-----|-----------|-------------------|---------------------|-------|
| #70 | 1.1659 | Int6+MLP3x+sliding | VALIDATED_SIGNIFICANT | HIGH | 100+ reproductions |
| #164 | 1.1524 | OrthoInit+BigramHash+SmearGate | VALIDATED_SIGNIFICANT | HIGH | Systematic stack |
| #198 | 1.1318 | 11L+WD=0.04+SWA | VALIDATED_SIGNIFICANT | HIGH | Community standard |
| #287 | 1.1271 | XSA+EMA | VALIDATED_SIGNIFICANT | HIGH | Multiple reproductions |
| #315 | 1.1248 | Partial RoPE+LN Scale | VALIDATED_SIGNIFICANT | HIGH | Dead code issue noted |
| #379 | 1.1257 | GPTQ-lite | VALIDATED_SIGNIFICANT | HIGH | 5+ reproductions |
| #442 | 1.1027 | AdamW TTT (vs SGD) | SINGLE_SEED_PENDING | LOW | p < 0.01 claimed, needs seeds |
| #466 | 1.1354 | BigramHash 12288 | VALIDATED_SIGNIFICANT | HIGH | 3-seed mean |
| #473 | 1.1214 | Legal score-first TTT | VALIDATED_SIGNIFICANT | HIGH | Multiple validations |
| #490 | 1.0891 | Value Residual+AdamW TTT | SINGLE_SEED_PENDING | LOW | Withdrawn? Needs validation |
| #507 | 1.1558 | Catalytic Residuals | MULTI_SEED_PENDING | MEDIUM | Novel, needs confirmation |
| #528 | 1.1195 | Full GPTQ+TTT | SINGLE_SEED_PENDING | LOW | Single seed |
| #532 | DQ | Codebook+Huffman | CATEGORY_1 | UNUSABLE | TTT non-compliant |
| #538 | 1.1511 | FP8+Arithmetic coding | MULTI_SEED_PENDING | MEDIUM | Novel |
| #544 | 1.1179 | Int5 GPTQ+TTT | SINGLE_SEED_PENDING | LOW | Single seed |

---

## Change Log (v1 → v2)

| Issue | Critique Ref | Change |
|-------|--------------|--------|
| Statistical threshold contradiction | A1 | Reconciled effect size and delta as alternative practical significance criteria; clarified they are not joint requirements |
| Mixed-seed results not handled | A2 | Added MIXED_SIGNALS to ValidationStatus; added signal_ratio reporting |
| Baseline requirements underspecified | A3 | Added baseline chaining rules; specified SINGLE_SEED_PENDING cannot validate multi-seed experiments |
| Prevention not implementable | B1 | Renamed to "Pre-flight Advisory"; scoped to config validation + code pattern matching; explicitly documented limitations |
| Log parser specification absent | B4 | Added StructuredExperimentLog schema; specified three input formats; documented conversion confidence |
| JSONL storage won't scale | A6 | Replaced with SQLite with proper indexes; added deduplication strategy |
| Missing ReproductionRecord schema | A5 | Added complete ReproductionRecord schema with confidence levels |
| No retrospective invalidation | C1 | Added InvalidationEvent schema and invalidation protocol with propagation rules |
| Phase timelines optimistic | C2 | Extended to 13 weeks total (3 + 6 + 4); Phase 2 now 6 weeks to properly scope log parser |
| Composite status undefined | C1 | Added composite_confidence() function combining validation and leakage status |
| Scale-dependent Category 2 | B2 | Added document count thresholds for Category 2 severity |
