# Failure Mode Knowledge Base — Implementation Plan v2
*For QKayV AI Agent — 2026-03-24*
*Addressing critique findings from `critique_failure_mode_kb.md`*

---

## Executive Summary

The Failure Mode Knowledge Base (FMKB) captures negative results with mechanistic explanations, enabling QKayV to warn researchers before they repeat known failure patterns. Unlike a simple list of "what failed," the FMKB stores the causal chain: technique → mechanism → triggering conditions → counterfactual.

**What changed from v1**: This v2 addresses six critical architectural flaws identified in the critique: (1) replaces string-based conditions with typed numeric/categorical fields, (2) adds full lifecycle management with entry_status and overturn protocols, (3) elevates partial failures to first-class schema elements, (4) specifies a coherent query engine architecture, (5) fixes incorrect entries FM-001 and FM-009, and (6) adds a damping mechanism for the FMKB↔Graph circular dependency.

---

## 1. What Constitutes a Failure Mode Entry

### 1.1 Core Fields

| Field | Description | Example |
|-------|-------------|---------|
| `id` | UUID | `"550e8400-e29b-41d4-a716-446655440001"` |
| `technique` | The technique attempted | `"Multi-Token Prediction (MTP)"` |
| `outcome` | Observable result (quantitative) | see Section 2.1 |
| `regime_specific_outcomes` | List of condition-outcome pairs for partial failures | see Section 2.1 |
| `root_cause_mechanism` | The causal mechanism behind the failure | `"28ms/step overhead consumed training budget"` |
| `triggering_conditions` | Typed condition fields (not natural language) | see Section 2.1 |
| `counterfactual` | What would need to change for success | see Section 2.1 |
| `entry_status` | Lifecycle state of this entry | `active` |
| `related_techniques` | Techniques with similar failure modes | `["MoE Routing", "QAT + EMA"]` |
| `severity` | Impact severity (1-5) | 4 |
| `confidence` | Confidence level (0.0-1.0) | 0.95 |

### 1.2 Severity Scale

| Score | Label | Description |
|-------|-------|-------------|
| 1 | Negligible | < 0.001 BPB impact or neutral result |
| 2 | Minor | 0.001 - 0.005 BPB impact |
| 3 | Moderate | 0.005 - 0.015 BPB impact |
| 4 | Major | 0.015 - 0.040 BPB impact |
| 5 | Catastrophic | > 0.040 BPB impact or training divergence |

### 1.3 Confidence Scoring

Confidence is derived from:
- **Multi-source validation**: Was this failure reproduced by multiple independent experiments? (+0.2 per independent source, max +0.4)
- **Ablation evidence**: Was the failure mode isolated via ablation? (+0.2)
- **Mechanism clarity**: Is the causal mechanism explicitly documented? (+0.1)
- **Condition specificity**: Are triggering conditions precisely defined? (+0.1)

**Formula**: `confidence = base(0.3) + src_validation + ablation + mechanism_clarity + condition_specificity`

### 1.4 Entry Status Lifecycle

Every entry has a lifecycle state to handle failure overturning:

```yaml
entry_status: enum[active, overturned, deleted, disputed]
```

| Status | Meaning | Behavior |
|--------|---------|----------|
| `active` | Entry is currently valid | Used in all queries |
| `overturned` | Entry was valid but is no longer applicable | Preserved for audit trail; excluded from queries by default |
| `deleted` | Entry was erroneous (e.g., bad source data) | Preserved for audit trail; excluded from queries |
| `disputed` | Conflicting evidence exists; entry under review | Included in queries with `confidence` penalized |

**Overturn protocol**:
1. New evidence contradicts an active entry → entry moves to `disputed`
2. If dispute is resolved in favor of overturning → entry moves to `overturned`
3. Overturn evidence is captured in `overturn_evidence` block (see Section 2.3)
4. FMKB emits an `EntryOverturned` event to the event bus
5. The event bus triggers graph edge retraction for any edges inferred from this entry

---

## 2. Data Model

### 2.1 Primary Schema (v2 — typed conditions)

```yaml
failure_mode_entry:
  id: string (UUID)
  created_at: timestamp
  updated_at: timestamp
  entry_status: enum[active, overturned, deleted, disputed]
  overturned_evidence: list[overturn_evidence_record] | null  # null if status != overturned

  technique: string
  technique_category: enum[architecture, quantization, optimizer, evaluation, test_time, regularization, speed, training]
  outcome: outcome_record
  regime_specific_outcomes: list[regime_outcome_record] | null  # null if not a partial failure

  mechanism:
    causal_chain: string  # Human-readable mechanistic explanation
    mechanism_type: enum[
      throughput_overhead,
      quantization_error_amplification,
      convergence_interference,
      mechanism_redundancy,
      dataset_mismatch,
      scale_mismatch,
      constraint_violation,
      implementation_defect,
      hw_arch_incompatibility
    ]
    throughput_impact_ms_per_step: float | null
    memory_impact_mb: float | null
    convergence_impact: string | null

  conditions: conditions_record  # Single-condition-block for simple failures

  experiment_context: experiment_context_record | null  # null if not relevant

  counterfactual:
    condition_for_success: string
    required_change: string
    expected_benefit_if_fixed: float | null
    success_conditions: typed_condition_record | null  # For query engine matching

  relationships:
    related_failures: list[string] (entry IDs)  # Denormalized; FK without enforcement
    conflicts_with: list[string] (technique names)
    requires: list[string] (prerequisite techniques)
    redundancies: list[{technique, mechanism_tag}]  # For Query D

  metadata:
    sources: list[{source_id, source_type, extraction_method, extraction_confidence}]
    confidence: float
    severity: integer
    tags: string[]
    notes: string | null

# ---- SUB-RECORDS ----

outcome_record:
  primary_metric:
    value: float
    unit: string  # "BPB", "perplexity", "tokens_per_sec", "compression_ratio", etc.
    baseline_value: float
    delta: float
    direction: enum[negative, positive, neutral]
  secondary_metrics: list[{name, value, unit, delta, direction}] | null

regime_outcome_record:  # For partial failures (first-class in v2)
  regime_id: string  # "regime_A", "regime_B", etc.
  conditions: typed_condition_record  # Matches query context
  outcome: outcome_record
  description: string  # Human-readable: "undertrained models (< 10K steps)"

typed_condition_record:  # Replaces all natural-language condition strings
  model_scale:
    min_params: integer | null
    max_params: integer | null
    min_inclusive: boolean
    max_inclusive: boolean
    min_exclusive: boolean
    max_exclusive: boolean
  training_steps:
    min_steps: integer | null
    max_steps: integer | null
    min_inclusive: boolean
    max_inclusive: boolean
  time_budget:
    min_minutes: integer | null
    max_minutes: integer | null
  memory_budget:
    min_mb: integer | null
    max_mb: integer | null
  quantization_precision:
    min_bits: integer | null  # e.g., 4
    max_bits: integer | null  # e.g., 8
    allowed_values: list[integer] | null  # e.g., [4, 5, 6, 8] for specific support
    model_component: enum[full_model, mlp_only, attention_only, embeddings, head] | null
  hardware: list[string] | null  # e.g., ["H100", "A100"]
  dataset_type: list[string] | null  # e.g., ["FineWeb", "Wikipedia"]
  dataset_size:
    min_tokens: integer | null
    max_tokens: integer | null
  architecture_family: enum[decoder_only, encoder_decoder, diffusion, hybrid] | null
  constraint_type:
    time_constrained: boolean
    time_budget_minutes: integer | null
    memory_constrained: boolean
    memory_budget_mb: integer | null
    compute_unconstrained: boolean
  hyperparameters:
    learning_rate: {min: float | null, max: float | null, exact: float | null} | null
    batch_size: {min: integer | null, max: integer | null, exact: integer | null} | null
    warmup_steps: {min: integer | null, max: integer | null} | null
    weight_decay: {min: float | null, max: float | null} | null
  random_seed_sensitivity: enum[low, medium, high, critical] | null

conditions_record:  # The conditions block in a standard (non-partial) entry
  applies_to: typed_condition_record  # Simplified: conditions always have typed fields
  prerequisite_techniques: list[string] | null  # e.g., ["requires int6 quantization"]
  incompatible_techniques: list[string] | null  # e.g., ["fails when combined with depth_recurrence"]

experiment_context_record:
  model_architecture:
    type: enum[decoder_only, encoder_decoder, diffusion, hybrid]
    num_layers: integer | null
    hidden_size: integer | null
    attention_type: string | null
    use_moe: boolean
    moe_num_experts: integer | null
  hyperparameters:
    learning_rate: float | null
    batch_size: integer | null
    warmup_steps: integer | null
    weight_decay: float | null
  dataset:
    size_tokens: integer | null
    avg_document_length_tokens: integer | null
    domain: string | null
  random_seed: integer | null  # For reproducibility assessment

overturn_evidence_record:
  source: string  # e.g., "PR #500", "User report"
  explanation: string
  date: timestamp
  submitted_by: string | null
```

### 2.2 Uncertainty Representation (v2)

```yaml
uncertainty:
  mechanism_confidence: enum[known, suspected, unknown]
  condition_boundary: enum[well_bounded, approximate, poorly_constrained]
  outcome_variance: float  # Standard deviation across experiments
  conflicting_evidence: boolean
  conflicting_sources: list[{source, evidence, weight}] | null
  temporal_boundary: boolean  # true if boundary may shift with new hardware
  hardware_generation_sensitivity: list[string] | null  # e.g., ["H100", "H200"]
```

### 2.3 Integration with Dependency Graph (v2 — with damping)

**Problem from v1**: FMKB infers graph edges → graph used for planning → planning generates new FMKB entries → infers more edges. No damping mechanism.

**Solution: Event-sourced edge inference with provenance and damping**

```yaml
# Every FMKB entry emits events, not direct mutations

fmkb_event:
  event_type: enum[EntryAdded, EntryUpdated, EntryOverturned, EntryDeleted]
  entry_id: string
  timestamp: timestamp
  payload:
    # For EntryAdded:
    inferred_edges: list[{
      edge_id: string (generated),
      from_node: string,
      to_node: string,
      edge_type: enum[requires, conflicts_with, redundant_with, tag关联],
      confidence: float,  # Derived from entry confidence
      provenance: enum[fmkb_inferred, curated],
      approval_status: enum[auto_approved, pending_review, rejected],
      auto_approval_threshold: float  # edges with confidence >= 0.85 are auto-approved
    }]

# Damping mechanism:
# 1. FMKB-inferred edges start at `pending_review` unless confidence >= 0.85
# 2. FMKB-inferred edges with provenance=fmkb_inferred have max_confidence=0.7
#    (never exceeds the FMKB entry confidence)
# 3. Curated edges (provenance=curated) have no confidence cap
# 4. If a curated edge contradicts an FMKB-inferred edge, the FMKB edge is demoted
#    to rejected and the entry status is moved to disputed
# 5. FMKB entry overturn → emit EdgeRetracted event → graph removes edge if
#    provenance=fmkb_inferred
```

**Edge lifecycle ownership**:

| Provenance | Auto-approved threshold | Max confidence | Can be overturned by curated? |
|------------|------------------------|-----------------|-------------------------------|
| `curated` | N/A (always manual) | 1.0 | N/A |
| `fmkb_inferred` | >= 0.85 | entry.confidence | Yes |
| `fmkb_inferred_approved` | N/A (manually approved) | 1.0 | Yes |

### 2.4 Partial Failures (v2 — first-class schema)

Partial failures are stored using `regime_specific_outcomes`, not as an aside:

```yaml
# Example: int5 MLP quantization — a partial failure
id: "FM-003-v2"
technique: "int5 MLP quantization"
entry_status: "active"
outcome:
  primary_metric:
    value: null  # No single outcome — use regime_specific_outcomes
    unit: "BPB"
    direction: "negative"  # Overall classification
regime_specific_outcomes:
  - regime_id: "undertrained"
    conditions:
      model_scale: {min_params: null, max_params: null}
      training_steps: {max_steps: 10000, max_inclusive: false}
      quantization_precision: {allowed_values: [5], model_component: mlp_only}
    outcome:
      primary_metric:
        value: +0.007
        unit: "BPB"
        delta: +0.007
        direction: "negative"
      description: "Undertrained models (< 10K steps) show significant penalty"
  - regime_id: "well_trained"
    conditions:
      model_scale: {min_params: null, max_params: null}
      training_steps: {min_steps: 15000, min_inclusive: true}
      quantization_precision: {allowed_values: [5], model_component: mlp_only}
    outcome:
      primary_metric:
        value: -0.001
        unit: "BPB"
        delta: -0.001
        direction: "neutral"
      description: "Well-trained models (>= 15K steps) show no penalty"
  - regime_id: "competitive"
    conditions:
      model_scale: {min_params: null, max_params: null}
      training_steps: {min_steps: 20000, min_inclusive: true}
      quantization_precision: {allowed_values: [5], model_component: mlp_only}
    outcome:
      primary_metric:
        value: 0.0
        unit: "BPB"
        delta: 0.0
        direction: "neutral"
      description: "Competitively trained models (20K+ steps) show no penalty"
mechanism:
  causal_chain: "int5 quantization error accumulates in MLP weights. Error is correctable during extended training but becomes permanent if training ends before convergence."
  mechanism_type: "quantization_error_amplification"
conditions:
  applies_to:  # Fallback conditions if no regime matches
    quantization_precision: {allowed_values: [5], model_component: mlp_only}
  prerequisite_techniques: null
  incompatible_techniques: null
# ... rest of entry
```

---

## 3. Knowledge Acquisition

### 3.1 Source Types and Extraction Priority

| Source | Format | Extraction Method | Priority |
|--------|--------|-------------------|----------|
| Research reports | Markdown | LLM-assisted extraction with structured prompts | P0 |
| PR descriptions | Markdown | LLM extraction with "Failure Pattern" system prompt | P1 |
| Experiment logs | JSON/structured | Automated extraction of negative results with thresholds | P1 |
| GitHub issues | Markdown | LLM extraction of failure reports | P2 |
| User feedback | Structured input | Form-based submission with validation | P2 |
| External ML literature | PDF/HTML | LLM extraction with domain-specific prompts | P3 |

### 3.2 LLM Extraction Pipeline

Before Phase 2, define extraction precision:

1. **Ground truth dataset**: Manually annotate 20 failure reports from parameter golf research
2. **Definition of precision**: `true_positives / (true_positives + false_positives)` where a false positive is a field that the LLM fills with a value that contradicts the source
3. **Measurement**: Run LLM extraction against ground truth; compute precision before building Phase 2

**Extraction prompt**:
```
You are analyzing a research report for failure mode entries.
For each failure you identify, extract structured fields using the FMKB v2 schema.
Use TYPED FIELDS for all conditions — no natural language ranges like "< 100M params".
Use numeric bounds: min_params, max_params, min_steps, max_steps, etc.
If a condition is unknown, use null.

Format each failure as a structured entry matching the FMKB v2 schema.
Flag entries as UNCERTAIN if mechanism or conditions are not explicit.
```

### 3.3 User Feedback Loop

1. **Inline warnings**: When QKayV suggests a technique, show FMKB warnings if any known failures involve that technique
2. **Feedback capture**: After each experiment, prompt user: "Did any technique behave unexpectedly?"
3. **Validation workflow**: New user-reported failures are flagged as `disputed` pending confirmation
4. **Community signal**: Multiple independent user reports on same failure mode increase confidence and can trigger `disputed` → `active`

---

## 4. Query Interface

### 4.1 Core Query Functions

#### Query A: "Will technique X fail given my setup?"

**Input**:
```yaml
query:
  technique: string
  context:
    hardware: string
    model_size_params: integer
    training_steps: integer
    time_budget_minutes: integer | null
    memory_budget_mb: integer | null
    dataset_type: string
    architecture_family: enum[decoder_only, encoder_decoder, diffusion, hybrid]
    active_techniques: list[string]
    hyperparameters:
      learning_rate: float | null
      batch_size: integer | null
```

**Output**:
```yaml
response:
  will_fail: boolean
  matching_regime: string | null  # regime_id if partial failure match
  failure_modes: list[{
    entry_id: string
    regime_id: string | null  # null for simple entries
    mechanism: string
    severity: integer
    confidence: float
    matching_conditions: list[string]  # Which typed conditions matched
    non_matching_conditions: list[string]  # Which conditions did not apply
    counterfactual: string
  }]
  safe_if: string | null
  alternative_suggestions: list[{technique, expected_delta, mechanism}]
```

#### Query B: "What is the most likely failure mode for this technique?"

**Input**: `technique: string, context: {...}`

**Output**: Single highest-confidence failure mode entry with mechanism explanation.

#### Query C: REMOVED from KB
Query C ("What throughput overhead threshold must a technique stay below?") is a derived calculation, not a KB lookup. The formula `max_allowed_overhead = baseline_throughput * (time_budget / reference_training_steps) - baseline_throughput` belongs in the **planner**, not the KB. The KB stores technique overhead measurements; the planner computes thresholds.

#### Query D: "Which techniques are known to have negative interactions?"

**Input**: `technique_a: string, technique_b: string`

**Output**:
```yaml
response:
  has_negative_interaction: boolean
  interaction_type: enum[same_mechanism_redundancy, overhead_amplification, convergence_interference, quant_error_amplification]
  combined_effect: float | null
  mechanism: string
  recommendation: enum[avoid_combination, use_sequentially, requires_alternation]
```

#### Query E: "Given my current stack, what should I avoid?"

**Input**: `active_techniques: list[string], context: {...}`

**Output**: List of techniques to avoid with specific failure mode entries and explanations.

### 4.2 Query Engine Architecture (v2 — coherent design)

**The v1 critique correctly identified that SQLite + FTS5 + LLM is not an architecture. Here is the v2 architecture.**

The query engine has three layers:

#### Layer 1: PostgreSQL with columnar indexes (primary store)

```sql
-- Core tables
CREATE TABLE failure_mode_entries (
  id UUID PRIMARY KEY,
  technique TEXT NOT NULL,
  technique_category TEXT,
  entry_status TEXT NOT NULL DEFAULT 'active',
  severity INTEGER NOT NULL,
  confidence REAL NOT NULL,
  outcome_json JSONB NOT NULL,  -- outcome_record
  regime_specific_outcomes_json JSONB,  -- list[regime_outcome_record], null if simple
  mechanism_json JSONB NOT NULL,
  conditions_json JSONB NOT NULL,  -- conditions_record (always typed)
  experiment_context_json JSONB,  -- experiment_context_record, null if generic
  counterfactual_json JSONB NOT NULL,
  relationships_json JSONB NOT NULL,
  uncertainty_json JSONB,
  metadata_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

-- Indexes for multi-dimensional condition matching
CREATE INDEX idx_fm_scale ON failure_mode_entries
  USING GIN ((conditions_json->'applies_to'->'model_scale'));
CREATE INDEX idx_fm_steps ON failure_mode_entries
  USING GIN ((conditions_json->'applies_to'->'training_steps'));
CREATE INDEX idx_fm_quant ON failure_mode_entries
  USING GIN ((conditions_json->'applies_to'->'quantization_precision'));
CREATE INDEX idx_fm_constraints ON failure_mode_entries
  USING GIN ((conditions_json->'applies_to'->'constraint_type'));
CREATE INDEX idx_fm_status ON failure_mode_entries (entry_status);
CREATE INDEX idx_fm_technique ON failure_mode_entries (technique);
CREATE INDEX idx_fm_severity ON failure_mode_entries (severity);
```

**Why PostgreSQL**:
- JSONB columns allow flexible regime_specific_outcomes without schema changes
- GIN indexes on JSONB fields support multi-dimensional range queries without O(n²) scans
- `entry_status` filter eliminates overturned/deleted entries at index time
- Can add `jsonb_path_query` for regime matching in a single SQL query

#### Layer 2: Condition Evaluator Service (not LLM)

The Condition Evaluator is a dedicated service that takes a technique + query context and returns which regime matches (if any):

```python
class ConditionEvaluator:
    def evaluate(self, query_context: QueryContext, entry: FailureModeEntry) -> MatchResult:
        """
        Returns MatchResult with:
          - matched_regime_id: str | None (None = simple entry, not partial)
          - matching_conditions: list[str]
          - non_matching_conditions: list[str]
          - match_score: float (0.0-1.0, weighted by severity of matching conditions)
        """
```

The Condition Evaluator does NOT use an LLM. It operates purely on typed fields:

```python
def evaluate_regime(regime: RegimeOutcomeRecord, ctx: QueryContext) -> bool:
    """Evaluate a single regime's typed conditions against query context."""
    cond = regime.conditions

    # Model scale check
    if cond.model_scale:
        scale = ctx.model_size_params
        if not bounds_check(scale, cond.model_scale):
            return False

    # Training steps check
    if cond.training_steps:
        steps = ctx.training_steps
        if not bounds_check(steps, cond.training_steps):
            return False

    # Quantization precision check
    if cond.quantization_precision:
        bits = ctx.quantization_bits
        component = ctx.quantization_component
        if not quant_check(bits, component, cond.quantization_precision):
            return False

    # ... other condition types

    return True

def bounds_check(value: int, bounds: NumericBounds) -> bool:
    """Evaluate typed numeric bounds without string parsing."""
    if bounds.min is not None:
        if bounds.min_inclusive and value < bounds.min: return False
        if bounds.min_exclusive and value <= bounds.min: return False
    if bounds.max is not None:
        if bounds.max_inclusive and value > bounds.max: return False
        if bounds.max_exclusive and value >= bounds.max: return False
    return True
```

The Condition Evaluator is a pure function — unit testable without an LLM or database.

#### Layer 3: LLM for freeform queries only

For "Explain why FM-006 fails" style queries, an LLM is used to generate natural language explanations from structured entries. The LLM is NOT used for condition matching.

**Layer responsibilities**:
- Layer 1 (PostgreSQL): Fast indexed filtering on typed fields, status, severity, technique
- Layer 2 (Condition Evaluator): Multi-dimensional condition matching for partial failures
- Layer 3 (LLM): Natural language explanation generation only

---

## 5. Integration with Dependency Graph

### 5.1 Event-Sourced Graph Updates

Instead of direct FMKB → Graph mutations, all updates flow through an event bus:

```
FMKB Entry Added
    ↓
Event: EntryAdded { entry_id, inferred_edges[] }
    ↓
Edge Approval Service
    ↓ (if confidence >= 0.85) → auto-approved
    ↓ (if confidence < 0.85) → queued for review
    ↓
Graph receives: EdgeAdded or EdgePendingReview
    ↓
Graph updates internal state
    ↓
Planning service consumes Graph state
    ↓
Planning generates new candidate techniques
    ↓
[Cycle continues but damped by approval queue]
```

### 5.2 Damping Mechanism

1. **Confidence cap on FMKB-inferred edges**: FMKB-inferred edges cannot exceed the entry's confidence score (max 0.95)
2. **Auto-approval threshold**: Only edges with confidence >= 0.85 bypass manual review
3. **Curated vs. inferred distinction**: Curated edges always win in conflicts
4. **Stale entry invalidation**: If an FMKB entry moves to `overturned`, all its inferred edges are retracted via `EdgeRetracted` event
5. **Edge confidence decay**: FMKB-inferred edges lose 0.05 confidence per quarter unless re-confirmed by new entries

### 5.3 Planning Algorithm with FMKB Pruning

```python
def plan_with_fmkb(candidates: list[Technique], context: QueryContext) -> PlanningResult:
    for candidate in candidates:
        # Query A: will this technique fail?
        result = query_engine.query_a(candidate, context)

        if result.will_fail:
            for failure in result.failure_modes:
                if failure.severity >= context.severity_threshold:
                    # Remove candidate
                    log_warning(f"Removed {candidate}: {failure.mechanism}")
                else:
                    # Include with warning
                    add_warning(candidate, failure)

        # Query D: check pairwise interactions
        for other in active_techniques:
            interaction = query_engine.query_d(candidate, other)
            if interaction.has_negative_interaction:
                if interaction.recommendation == Recommendation.AVOID_COMBINATION:
                    remove_conflicting(candidate, other)

    return pruned_candidates

# Severity threshold is configurable per-user, not hardcoded to 3
# Default: 3 (moderate or worse blocks adoption)
```

---

## 6. Specific Failures (v2 — corrected)

### FM-001: MTP (v2 — fixed condition representation)

**v1 critique**: `scale_range: "< 100M params"` is misleading — MTP fails due to time-constrained training, not model scale.

```yaml
id: "FM-001-v2"
technique: "Multi-Token Prediction (MTP)"
technique_category: "architecture"
entry_status: "active"
outcome:
  primary_metric:
    value: +0.028
    unit: "BPB"
    baseline_value: 1.1248
    delta: +0.028
    direction: "negative"
regime_specific_outcomes: null  # Not a partial failure
mechanism:
  causal_chain: "MTP adds 28ms/step overhead. At 86ms/step baseline, this reduces training steps in 10-minute budget from ~6,977 to ~5,840. The lost gradient updates cost more BPB than MTP's improved gradient signal gains."
  mechanism_type: "throughput_overhead"
  throughput_impact_ms_per_step: 28
  memory_impact_mb: null
  convergence_impact: "reduced training steps"
conditions:
  applies_to:
    model_scale: {min_params: null, max_params: null}  # No scale dependence
    training_steps: {min_params: null, max_params: null}  # Any scale
    time_budget:
      max_minutes: 10  # Time-constrained regimes
    constraint_type:
      time_constrained: true
      time_budget_minutes: 10
      memory_constrained: false
      compute_unconstrained: false
    hardware: ["H100"]  # Specific to H100 measured overhead
  prerequisite_techniques: null
  incompatible_techniques: null
experiment_context: null  # Generic — not architecture-specific
counterfactual:
  condition_for_success: "Overhead must be < 5ms/step OR training must be compute-unconstrained (no time budget)"
  required_change: "Optimize MTP implementation to reduce per-step overhead"
  expected_benefit_if_fixed: -0.010
  success_conditions:
    throughput_impact_ms_per_step: {max: 5}
relationships:
  related_failures: ["FM-002", "FM-008"]
  conflicts_with: ["time_constrained_training"]
  requires: []
metadata:
  sources: [{source_id: "PR #375", source_type: "pr", extraction_method: "manual", extraction_confidence: 0.95}]
  confidence: 0.95
  severity: 4
  tags: ["throughput", "overhead", "time-budget"]
  notes: "MTP is not fundamentally flawed; the failure is purely an implementation overhead issue in the time-constrained regime. Works in compute-unconstrained settings."
```

### FM-003: int5 MLP (v2 — first-class partial failure)

```yaml
id: "FM-003-v2"
technique: "int5 MLP quantization"
technique_category: "quantization"
entry_status: "active"
outcome:
  primary_metric:
    value: null  # Partial failure — no single outcome
    unit: "BPB"
    direction: "negative"
regime_specific_outcomes:
  - regime_id: "undertrained"
    conditions:
      training_steps: {max_steps: 10000, max_inclusive: false}
    outcome:
      primary_metric:
        value: +0.007
        unit: "BPB"
        delta: +0.007
        direction: "negative"
      description: "Undertrained models (< 10K steps): significant penalty"
  - regime_id: "well_trained"
    conditions:
      training_steps: {min_steps: 15000, min_inclusive: true}
    outcome:
      primary_metric:
        value: -0.001
        unit: "BPB"
        delta: -0.001
        direction: "neutral"
      description: "Well-trained models (>= 15K steps): no penalty"
  - regime_id: "competitive"
    conditions:
      training_steps: {min_steps: 20000, min_inclusive: true}
    outcome:
      primary_metric:
        value: 0.0
        unit: "BPB"
        delta: 0.0
        direction: "neutral"
      description: "Competitively trained models (20K+ steps): no penalty"
mechanism:
  causal_chain: "int5 quantization error accumulates in MLP weights. Error is correctable during extended training but becomes permanent if training ends before convergence."
  mechanism_type: "quantization_error_amplification"
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "quantization error accumulation in undertrained models"
conditions:
  applies_to:
    quantization_precision: {allowed_values: [5], model_component: mlp_only}
  prerequisite_techniques: null
  incompatible_techniques: null
experiment_context: null
counterfactual:
  condition_for_success: "Use int6 for MLP weights; int5 acceptable only for models trained >= 15K steps"
  required_change: "Use int6 for MLP; int5 is context-dependent"
  expected_benefit_if_fixed: null
  success_conditions:
    training_steps: {min_steps: 15000}
relationships:
  related_failures: ["FM-004"]
  conflicts_with: ["undertrained_model"]
  requires: []
metadata:
  sources: [{source_id: "PR #480", source_type: "pr", extraction_method: "manual", extraction_confidence: 0.85}]
  confidence: 0.85
  severity: 3
  tags: ["quantization", "precision", "mlp", "partial_failure"]
  notes: "Partial failure: outcome depends on training duration. Integrated as first-class schema."
```

### FM-006: Depth Recurrence + int6 (v2 — correct conflict relationships)

```yaml
id: "FM-006-v2"
technique: "Depth Recurrence + int6 Quantization"
technique_category: "architecture"
entry_status: "active"
outcome:
  primary_metric:
    value: +1.14
    unit: "BPB"
    baseline_value: 1.1248
    delta: +1.14
    direction: "negative"
regime_specific_outcomes: null
mechanism:
  causal_chain: "Weight sharing across 3 recurrence cycles amplifies int6 quantization error ~900x. Each recurrence passes the quantization error forward, compounding across cycles."
  mechanism_type: "quantization_error_amplification"
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "quantization error amplification"
conditions:
  applies_to:
    quantization_precision: {min_bits: null, max_bits: null, allowed_values: [6], model_component: full_model}
    architecture_family: decoder_only  # Depth recurrence is decoder-only
  prerequisite_techniques: ["depth_recurrence", "int6_quantization"]
  incompatible_techniques: null
experiment_context: null
counterfactual:
  condition_for_success: "Use int8 or higher precision when combining with depth recurrence"
  required_change: "Depth recurrence is incompatible with int6; requires int8+ precision"
  expected_benefit_if_fixed: -0.010
  success_conditions:
    quantization_precision: {min_bits: 8}
relationships:
  related_failures: ["FM-004"]
  conflicts_with: ["int6_quantization", "int5_quantization"]
  redundancies: []  # v1 had this empty — correctly filled in v2
  requires: ["int8_or_higher"]
metadata:
  sources: [{source_id: "PR #363", source_type: "pr", extraction_method: "manual", extraction_confidence: 0.95}]
  confidence: 0.95
  severity: 5
  tags: ["quantization", "architecture", "error_amplification", "catastrophic"]
  notes: "Depth recurrence without quantization works (~1.177 BPB); failure is specific to the int6 combination"
```

### FM-007: XSA + TTT (v2 — fixed empty conflicts_with)

**v1 critique**: `conflicts_with: []` despite notes saying they "conflict" mechanistically.

```yaml
id: "FM-007-v2"
technique: "Exclusive Self Attention (XSA) + Test-Time Training (TTT)"
technique_category: "architecture"
entry_status: "active"
outcome:
  primary_metric:
    value: -0.016
    unit: "BPB"
    baseline_value: 1.1248
    delta: -0.016
    direction: "negative"
regime_specific_outcomes: null
mechanism:
  causal_chain: "XSA and TTT both target local context modeling. XSA removes self-value bias to force the model to attend to other tokens. TTT fine-tunes on local context patterns. Combined, they create redundancy where each technique's benefit is cancelled by the other's mechanism."
  mechanism_type: "mechanism_redundancy"
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "mechanism redundancy"
conditions:
  applies_to:
    model_scale: {min_params: null, max_params: null}
    training_steps: {min_steps: null, max_steps: null}
  prerequisite_techniques: ["XSA", "TTT"]
  incompatible_techniques: null
experiment_context: null
counterfactual:
  condition_for_success: "Use XSA OR TTT, but not both in same stack"
  required_change: "Choose one: XSA provides architectural improvement; TTT provides test-time adaptation"
  expected_benefit_if_fixed: null
  success_conditions:
    incompatible_techniques: ["XSA", "TTT"]  # Cannot have both
relationships:
  related_failures: []
  conflicts_with: ["XSA", "TTT"]  # v2: correctly populated
  redundancies: [{technique: "XSA", mechanism_tag: "local_context_modeling"},
                 {technique: "TTT", mechanism_tag: "local_context_modeling"}]
  requires: []
metadata:
  sources: [{source_id: "PR #303", source_type: "pr", extraction_method: "manual", extraction_confidence: 0.90}]
  confidence: 0.90
  severity: 3
  tags: ["redundancy", "mechanism_conflict", "local_context"]
  notes: "Individual XSA is positive (-0.005 BPB); individual TTT is positive (-0.010 to -0.020 BPB); combined is negative"
```

### FM-009: Label Smoothing — REMOVED from failures, relocated to neutral results

**v1 critique**: FM-009 is `direction: "neutral"` and `severity: 1` — it is not a failure. A "Failure Mode Knowledge Base" should not contain neutral results.

Label Smoothing (delta = 0.0 BPB, neutral) is moved to a separate **Neutral Results Log** with the same schema but filtered out of failure queries by default.

```yaml
# Separate entry type: neutral_result_entry
id: "NR-001"
result_type: "neutral"  # enum[positive, negative, neutral]
technique: "Label Smoothing"
technique_category: "regularization"
entry_status: "active"
outcome:
  primary_metric:
    value: 0.0
    unit: "BPB"
    delta: 0.0
    direction: "neutral"
mechanism:
  causal_chain: "Label smoothing provides no benefit at this model scale and dataset. The model is already well-regularized by other techniques (weight decay, EMA). The additional regularization from label smoothing is redundant."
  mechanism_type: "no_op"  # New mechanism type for neutral results
conditions:
  applies_to:
    model_scale: {min_params: null, max_params: 100_000_000, max_inclusive: false}
  prerequisite_techniques: null
  incompatible_techniques: null
experiment_context: null
counterfactual:
  condition_for_success: "May provide benefit on larger models or with less regularization"
  required_change: "None needed; this is a neutral result"
  expected_benefit_if_fixed: null
  success_conditions: null
relationships:
  related_failures: ["NR-002"]  # Links to other neutral results
  conflicts_with: []
  requires: []
metadata:
  sources: [{source_id: "PR #375", source_type: "pr", extraction_method: "manual", extraction_confidence: 0.85}]
  confidence: 0.85
  severity: 1  # Negligible — neutral result
  tags: ["regularization", "no_effect", "neutral_result"]
  notes: "Neutral result, not a failure. Stored in Neutral Results Log."
```

---

## 7. Implementation Roadmap (v2)

### Phase 1: Core Infrastructure (Weeks 1-3)

**Goal**: Build the FMKB v2 data model and basic query API with typed conditions.

**Tasks**:
1. Implement PostgreSQL schema with JSONB columns and GIN indexes for typed conditions
2. Build Condition Evaluator service (pure function, unit-testable, no LLM)
3. Implement entry lifecycle: `active`, `overturned`, `deleted`, `disputed` with overturn evidence capture
4. Build regime matching for partial failures (first-class in v2)
5. Build event emission for graph updates (EntryAdded, EntryOverturned, EdgeRetracted)
6. Port the 10 corrected failure entries from v1 to v2 schema
7. Move Label Smoothing and L1 Regularization to Neutral Results Log

**Deliverables**:
- FMKB v2 database with 10+ seed entries in typed schema
- Condition Evaluator service (unit tested)
- Event bus integration for graph updates
- Query API (A, B, D, E) with < 50ms latency for simple entries

**Success criteria**:
- Query A correctly matches partial failure regimes (int5 MLP: undertrained vs. well-trained)
- Entry overturn moves status and emits EdgeRetracted event
- Query latency < 50ms for single-technique lookup on 500 entries

### Phase 2: LLM-Assisted Extraction (Weeks 4-6)

**Goal**: Automatically extract failure modes from new experiment results with measurable precision.

**Tasks**:
1. Build ground truth test set: manually annotate 20 failure reports from parameter golf research
2. Define extraction precision: `tp / (tp + fp)` per field
3. Build LLM extraction pipeline with typed field output (not string ranges)
4. Measure LLM extraction precision against ground truth before production deployment
5. Implement user feedback capture with `disputed` → `active` workflow
6. Add provenance to all fields: `extraction_method`, `extraction_confidence`

**Deliverables**:
- Ground truth dataset with 20 annotated failure reports
- LLM extraction pipeline with measured precision >= 0.80
- User feedback workflow (report → dispute → validate → confirm/overturn)

**Success criteria**:
- LLM extraction precision >= 0.80 on ground truth test set
- New experiment failures extracted and entered within 1 hour of PR
- User confirmation reduces false positive rate by > 50%

### Phase 3: Graph Integration with Damping (Weeks 7-9)

**Goal**: Implement event-sourced graph updates with damping mechanism.

**Tasks**:
1. Build Edge Approval Service: auto-approve edges with confidence >= 0.85
2. Implement FMKB-inferred edge confidence cap (max = entry confidence)
3. Build EdgeRetracted event handler: when FMKB entry is overturned, retract inferred edges
4. Implement confidence decay for FMKB-inferred edges (0.05 per quarter)
5. Build curated vs. inferred conflict resolution: curated edges always win
6. Integrate with planning algorithm: severity threshold becomes configurable

**Deliverables**:
- Edge Approval Service with auto/manual approval queue
- Event bus connecting FMKB events to graph updates
- Configurable severity threshold for planning algorithm

**Success criteria**:
- FMKB-inferred edges cannot exceed entry confidence
- Overturned entry triggers EdgeRetracted event within 1 minute
- Circular dependency is damped: no edge confidence spiral

### Phase 4: Knowledge Base Expansion (Ongoing)

**Goal**: Grow FMKB beyond parameter golf to general ML training knowledge.

**Tasks**:
1. Extract failures from external ML literature (arXiv, papers with code)
2. Build community contribution workflow with review process
3. Add temporal awareness: failures tied to specific hardware generations
4. Cross-domain failure linking (similar failures in different technique categories)

**Deliverables**:
- External literature extraction pipeline
- Community contribution portal with contribution guidelines
- Hardware-generation-tagged entries (e.g., H100-specific failures)

---

## Appendix: Key Changes from v1 to v2

| Issue | v1 | v2 |
|-------|----|----|
| Condition representation | String fields (`"< 100M params"`) | Typed fields (`min_params`, `max_params`, `min_inclusive`, etc.) |
| Constraint type | Single enum | Multi-flag record with budget values |
| Partial failures | Aside (Section 3.3) | First-class `regime_specific_outcomes` in primary schema |
| Entry lifecycle | Not present | `entry_status` with `overturned_evidence` and protocol |
| Query engine | SQLite + FTS5 + LLM (incoherent) | PostgreSQL (layer 1) + Condition Evaluator (layer 2) + LLM for explanation only (layer 3) |
| FMKB-Graph circular dependency | Not addressed | Event-sourced with damping (edge approval, confidence cap, decay) |
| FM-001 (MTP) | `scale_range: "< 100M params"` (misleading) | `constraint_type: {time_constrained: true}` (correct) |
| FM-003 (int5 MLP) | Single outcome + aside note | First-class partial failure with regime_outcome_record |
| FM-007 (XSA + TTT) | `conflicts_with: []` (empty — wrong) | `conflicts_with: ["XSA", "TTT"]` + `redundancies` correctly populated |
| FM-009 (Label Smoothing) | In failure section with severity: 1 | Moved to Neutral Results Log |
| Query C | Listed as KB query (wrong) | Removed — belongs in planner as derived calculation |
| Severity threshold | Hardcoded 3 | Configurable per-user |

---

## Appendix: Technology Stack (v2)

- **Primary Database**: PostgreSQL 15+ with JSONB
  - GIN indexes on typed condition JSONB fields for multi-dimensional range queries
  - `jsonb_path_query` for regime matching
- **Condition Evaluator**: Python service (pure function, no LLM dependency)
- **Event Bus**: Internal message queue for FMKB → Graph events
- **LLM**: Used only for freeform explanation generation and initial extraction (not condition matching)
- **API**: REST API for QKayV core to query FMKB
