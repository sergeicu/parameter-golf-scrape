# Failure Mode Knowledge Base — Implementation Plan
*For QKayV AI Agent — 2026-03-24*

---

## Executive Summary

The Failure Mode Knowledge Base (FMKB) captures negative results with mechanistic explanations, enabling QKayV to warn researchers before they repeat known failure patterns. Unlike a simple list of "what failed," the FMKB stores the causal chain: technique → mechanism → triggering conditions → counterfactual. This transforms anecdotal failure reports into actionable intelligence for experiment planning.

---

## 1. What Constitutes a Failure Mode Entry

Each failure mode entry is a structured record with the following fields:

### 1.1 Core Fields

| Field | Description | Example |
|-------|-------------|---------|
| `technique` | The technique that was attempted | "Multi-Token Prediction (MTP)" |
| `outcome` | Observable result (quantitative) | "+0.028 BPB degradation" |
| `outcome_delta` | Numeric delta from baseline | `+0.028` |
| `outcome_unit` | Unit of measurement | "BPB" |
| `root_cause_mechanism` | The causal mechanism behind the failure | "28ms/step overhead consumed training budget, net negative" |
| `triggering_conditions` | When this failure occurs | "only fails under time-constrained training; works in compute-unconstrained settings" |
| `counterfactual` | What would need to change for success | "would work if overhead < 5ms/step" |
| `related_techniques` | Techniques with similar failure modes | ["MoE routing", "QAT + EMA"] |
| `severity` | Impact severity (1-5) | 4 |
| `confidence` | Confidence level in this failure mode (0.0-1.0) | 0.95 |

### 1.2 Severity Scale

| Score | Label | Description |
|-------|-------|-------------|
| 1 | Negligible | < 0.001 BPB impact |
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

---

## 2. Data Model

### 2.1 Primary Schema

```yaml
failure_mode_entry:
  id: string (UUID)
  created_at: timestamp
  updated_at: timestamp
  technique: string
  technique_category: enum[architecture, quantization, optimizer, evaluation, test_time, regularization, speed]
  outcome:
    value: float
    unit: string
    baseline_value: float
    delta: float
    direction: enum[negative, positive, neutral]
  mechanism:
    causal_chain: string  # Human-readable explanation
    throughput_impact_ms_per_step: float | null
    memory_impact_mb: float | null
    convergence_impact: string | null
  conditions:
    hardware_required: string[] | null
    scale_range: string | null  # e.g., "< 100M params", "> 1B params"
    training_duration: string | null  # e.g., "< 10K steps"
    constraint_type: enum[time_constrained, size_constrained, memory_constrained, compute_unconstrained]
    dataset_type: string | null
    prerequisites: string[]  # e.g., ["requires int6 quantization"]
  counterfactual:
    condition_for_success: string
    required_change: string
    expected_benefit_if_fixed: float | null
  relationships:
    related_failures: string[] (IDs)
    conflicts_with: string[] (technique names that fail when combined)
    requires: string[] (prerequisite techniques that must work first)
  metadata:
    source_pr: string
    source_issue: string | null
    confidence: float
    severity: integer
    tags: string[]
    notes: string | null
```

### 2.2 Uncertainty Representation

For entries where mechanism or conditions are uncertain:

```yaml
uncertainty:
  mechanism_confidence: enum[known, suspected, unknown]
  condition_boundary: enum[well_bounded, approximate, poorly_constrained]
  outcome_variance: float  # Standard deviation across experiments
  conflicting_evidence: boolean
  conflicting_sources: string[] | null
```

### 2.3 Integration with Dependency Graph

Failures often reveal hidden preconditions. The FMKB must interface with QKayV's dependency graph:

```yaml
graph_inference:
  inferred_prerequisites: string[]  # e.g., ["int6 quantization" inferred from MTP failure context
  inferred_conflicts: string[]  # e.g., ["depth recurrence"] conflicts with ["int6 quantization"]
  suggested_edges_to_add: list[{from, to, type, confidence}]
  suggested_nodes_to_add: list[{id, type, label}]
```

**Example**: MTP's failure revealed that time-constrained training is a precondition. This should add a "training_budget_type" node to the dependency graph with an edge to MTP.

---

## 3. Knowledge Acquisition

### 3.1 Source Types and Extraction Priority

| Source | Format | Extraction Method | Priority |
|--------|--------|-------------------|----------|
| Research reports (parameter_golf_research_report.md) | Markdown | LLM-assisted extraction with structured prompts | P0 (immediate) |
| PR descriptions | Markdown | LLM extraction with "Failure Pattern" system prompt | P1 |
| Experiment logs | JSON/structured | Automated extraction of negative results with thresholds | P1 |
| GitHub issues | Markdown | LLM extraction of failure reports | P2 |
| User feedback | Structured input | Form-based submission with validation | P2 |
| External ML literature | PDF/HTML | LLM extraction with domain-specific prompts | P3 |

### 3.2 LLM Extraction Prompt for Research Reports

```
You are analyzing a research report for failure mode entries.
For each failure you identify, extract:
1. Technique name
2. Quantitative outcome (value, unit, baseline, delta)
3. Root cause mechanism (explicit statement of WHY it failed)
4. Triggering conditions (hardware, scale, duration, constraints)
5. Counterfactual (what would need to change)
6. Related techniques with similar failures
7. Confidence assessment (multi-source? ablation? mechanism clear?)

Format each failure as a structured entry matching the FMKB schema.
Flag entries as UNCERTAIN if mechanism or conditions are not explicit.
```

### 3.3 Handling Partial Failures

Partial failures (works in some conditions, fails in others) require **conditional entries**:

```yaml
partial_failure_entry:
  technique: "int5 MLP quantization"
  conditions:
    - context: "well-trained models (> 15K steps)"
      outcome: "-0.001 BPB (neutral to positive)"
    - context: "undertrained models (< 10K steps)"
      outcome: "+0.007 BPB gap (significant penalty)"
  mechanism: "int5 quantization error accumulates during incomplete convergence"
  boundary_conditions:
    minimum_steps_for_success: 12000
    recommended_steps: 15000
    notes: "Competitively trained models at 20K steps show no penalty"
```

### 3.4 User Feedback Loop

1. **Inline warnings**: When QKayV suggests a technique, show FMKB warnings if any known failures involve that technique
2. **Feedback capture**: After each experiment, prompt user: "Did any technique behave unexpectedly?"
3. **Validation workflow**: New user-reported failures are flagged as UNCERTAIN pending confirmation
4. **Community signal**: Multiple independent user reports on same failure mode increase confidence

---

## 4. Query Interface

### 4.1 Core Query Functions

#### Query A: "Will technique X fail given my setup?"

**Input**:
```yaml
query:
  technique: string
  constraints:
    hardware: string
    training_duration_steps: integer
    time_budget_minutes: integer | null
    memory_budget_mb: integer | null
    model_size_params: integer
    dataset_type: string
  active_techniques: string[]  # What is already in the stack
```

**Output**:
```yaml
response:
  will_fail: boolean
  failure_modes: list[{
    entry_id: string
    mechanism: string
    severity: integer
    confidence: float
    matching_conditions: string[]  # Which query conditions triggered this
    non_matching_conditions: string[]  # Which conditions don't apply
    counterfactual: string
  }]
  safe_if: string | null  # Conditions under which technique would work
  alternative_suggestions: list[{technique, expected_delta, mechanism}]
```

#### Query B: "What is the most likely failure mode for this technique?"

**Input**: `technique: string, context: {scale, constraints, dataset}`

**Output**: Single highest-confidence failure mode entry with mechanism explanation.

#### Query C: "What throughput overhead threshold must a technique stay below?"

**Input**: `training_duration_steps: integer, time_budget_minutes: integer`

**Output**:
```yaml
response:
  max_allowed_overhead_ms_per_step: float
  calculation: "At 86ms/step baseline, 10-minute budget = 6,977 steps.
                 To match 20K-step reference, must stay under 5ms/step overhead"
  techniques_at_risk: list[{technique, measured_overhead_ms_per_step, risk_level}]
```

#### Query D: "Which techniques are known to have negative interactions?"

**Input**: `technique_a: string, technique_b: string`

**Output**:
```yaml
response:
  has_negative_interaction: boolean
  interaction_type: enum[same_mechanism_redundancy, overhead_amplification, convergence_interference, quant_error_amplification]
  combined_effect: float  # Expected delta when combined
  mechanism: string
  recommendation: enum[avoid_combination, use_sequentially, requires_alternation]
```

#### Query E: "Given my current stack, what should I avoid?"

**Input**: `active_techniques: string[], constraints: {...}`

**Output**: List of techniques to avoid with specific failure mode entries and explanations.

### 4.2 Query Engine Implementation

- **Storage**: SQLite with FTS5 for text search, structured indexes on technique, conditions, severity
- **Reasoning**: Rules engine + LLM for condition matching against query context
- **Fallback**: LLM-based freeform query when structured queries don't match

---

## 5. Integration with Dependency Graph

### 5.1 How Failures Update the Graph

Failures reveal hidden dependencies and conflicts that are not apparent from technique descriptions alone.

#### Pattern 1: Failure reveals prerequisite constraints

**Example**: MTP failed because training budget was constrained.

**Graph update**:
```
Add node: "training_budget_type" [constraint]
Add edge: "MTP" --requires--> "training_budget_type: time_unconstrained"
Add edge: "training_budget_type: time_constrained" --disables--> "MTP"
```

#### Pattern 2: Failure reveals quantization interactions

**Example**: Depth recurrence + int6 = 900x quant error amplification.

**Graph update**:
```
Add edge: "depth_recurrence" --conflicts_with--> "int6_quantization"
Add edge: "depth_recurrence" --requires--> "int8_or_higher_quantization"
Add note: "quantization_precision" is a hidden precondition for depth_recurrence
```

#### Pattern 3: Failure reveals mechanism redundancy

**Example**: XSA + TTT = -0.016 BPB (both target local context modeling).

**Graph update**:
```
Add edge: "XSA" --redundant_with--> "TTT"
Add edge: "XSA" --tag: "local_context_modeling"--> "mechanism_tag"
Add edge: "TTT" --tag: "local_context_modeling"--> "mechanism_tag"
```

### 5.2 Using FMKB to Prune the Dependency Graph During Planning

When QKayV plans an experiment stack:

1. **Conflict detection**: For each proposed technique pair, check FMKB for negative interaction entries
2. **Precondition inference**: For each technique, check if FMKB reveals hidden prerequisites
3. **Overhead budget check**: Calculate max allowed overhead from training constraints, flag techniques that exceed threshold
4. **Scale-gating**: Filter techniques that are known to fail at the current model scale

**Planning algorithm with FMKB pruning**:
```
1. Generate candidate technique list for desired improvement
2. For each candidate:
   a. Query FMKB: will_fail(given_current_stack + candidate, constraints)
   b. If failure_modes with severity >= 3: remove candidate, log warning
   c. If failure_modes with severity < 3: include with warning
3. Check pairwise interactions via Query D
4. Remove/conflict techniques with negative interactions
5. Return pruned candidate list with FMKB warnings attached
```

### 5.3 Graph Update Triggers

| Trigger | Graph Update |
|---------|--------------|
| New failure mode entry added | Infer and add prerequisite/conflict edges |
| Existing entry confidence increased | Upgrade inferred edges from SUSPECTED to VALIDATED |
| User reports unexpected success | Add exception case to triggering conditions |
| New technique combination validated | Add synergies to graph |

---

## 6. Specific Failures to Document from Parameter Golf

The following failures are extracted from the parameter golf research report. Each is formatted as a complete FMKB entry.

### Failure 1: Multi-Token Prediction (MTP)

```yaml
id: "FM-001"
technique: "Multi-Token Prediction (MTP)"
technique_category: "architecture"
outcome:
  value: +0.028
  unit: "BPB"
  baseline_value: 1.1248
  delta: +0.028
  direction: "negative"
mechanism:
  causal_chain: "MTP adds 28ms/step overhead. At 86ms/step baseline, this reduces training steps in 10-minute budget from ~6,977 to ~5,840. The lost gradient updates cost more BPB than MTP's improved gradient signal gains."
  throughput_impact_ms_per_step: 28
  memory_impact_mb: null
  convergence_impact: "reduced training steps"
conditions:
  hardware_required: ["H100"]
  scale_range: "< 100M params"
  training_duration: "10,000 steps (time-constrained)"
  constraint_type: "time_constrained"
  dataset_type: null
  prerequisites: []
counterfactual:
  condition_for_success: "Overhead must be < 5ms/step OR training must be compute-unconstrained"
  required_change: "Optimize MTP implementation to reduce per-step overhead"
  expected_benefit_if_fixed: -0.010  # theoretical, if overhead eliminated
relationships:
  related_failures: ["FM-002", "FM-008"]
  conflicts_with: ["time_constrained_training"]
  requires: []
metadata:
  source_pr: "PR #375"
  confidence: 0.95
  severity: 4
  tags: ["throughput", "overhead", "time-budget"]
  notes: "MTP is not fundamentally flawed; the failure is purely an implementation overhead issue in the time-constrained regime"
```

### Failure 2: QAT + EMA Combined

```yaml
id: "FM-002"
technique: "Quantization-Aware Training (QAT) + EMA"
technique_category: "quantization"
outcome:
  value: +0.018
  unit: "BPB"
  baseline_value: 1.1248
  delta: +0.018
  direction: "negative"
mechanism:
  causal_chain: "QAT costs 8% throughput; naive EMA implementation (cloning full state dict to CPU every step) costs 32% throughput. Combined overhead reduces steps below threshold needed to overcome the quality gains from both techniques."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "throughput degradation"
conditions:
  hardware_required: null
  scale_range: null
  training_duration: null
  constraint_type: "time_constrained"
  dataset_type: null
  prerequisites: ["QAT", "EMA"]
counterfactual:
  condition_for_success: "Either QAT throughput cost < 2% OR EMA implementation uses tensor-only tracking (not full state dict clone)"
  required_change: "Implement EMA without full state dict cloning; use tensor-only state tracking"
  expected_benefit_if_fixed: -0.005  # Combined would yield ~0.003 (EMA) + QAT quant benefit
relationships:
  related_failures: ["FM-001", "FM-008"]
  conflicts_with: ["naive_ema_implementation", "time_constrained_training"]
  requires: []
metadata:
  source_pr: "PR #360"
  confidence: 0.90
  severity: 4
  tags: ["throughput", "overhead", "implementation", "combined"]
  notes: "Individual QAT and EMA are both positive; their combination fails due to overhead multiplication"
```

### Failure 3: int5 MLP (Well-Trained Models)

```yaml
id: "FM-003"
technique: "int5 MLP quantization"
technique_category: "quantization"
outcome:
  value: +0.007
  unit: "BPB"
  baseline_value: 1.1248
  delta: +0.007
  direction: "negative"
mechanism:
  causal_chain: "int5 quantization error accumulates in MLP weights even in well-trained models. The 32 discrete levels are insufficient to capture the full weight distribution, creating reconstruction error that compounds through layers."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: null
conditions:
  hardware_required: null
  scale_range: null
  training_duration: "10,000-20,000 steps"
  constraint_type: "size_constrained"
  dataset_type: null
  prerequisites: ["int5_quantization"]
counterfactual:
  condition_for_success: "Models must be trained for > 25K steps OR use int6 for MLP instead"
  required_change: "Use int6 for MLP weights; int5 acceptable only for undertrained models"
  expected_benefit_if_fixed: null
relationships:
  related_failures: ["FM-004"]
  conflicts_with: ["undertrained_model"]
  requires: []
metadata:
  source_pr: "PR #480, PR #238"
  confidence: 0.85
  severity: 3
  tags: ["quantization", "precision", "mlp"]
  notes: "This is a partial failure — int5 MLP is acceptable for undertrained models but shows +0.007 BPB gap in competitive (20K step) training"
```

### Failure 4: int4 Quantization

```yaml
id: "FM-004"
technique: "int4 quantization (full model)"
technique_category: "quantization"
outcome:
  value: +0.065
  unit: "BPB"
  baseline_value: 1.1248
  delta: +0.065
  direction: "negative"
mechanism:
  causal_chain: "4-bit quantization (16 levels) is too aggressive for transformer weights. The quantization error exceeds the model's error correction capacity, resulting in catastrophic quality degradation. Error accumulates across 11 layers of forward pass."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "catastrophic quality degradation"
conditions:
  hardware_required: null
  scale_range: null
  training_duration: null
  constraint_type: "size_constrained"
  dataset_type: null
  prerequisites: []
counterfactual:
  condition_for_success: "Use int6 or int5 at minimum; int4 requires specialized training (e.g., BitNet b1.58 with adapted architecture)"
  required_change: "Do not use standard int4 on standard transformers; use BitNet-style ternary/binary architectures instead"
  expected_benefit_if_fixed: null
relationships:
  related_failures: ["FM-003"]
  conflicts_with: ["standard_transformer_architecture"]
  requires: []
metadata:
  source_pr: "PR #375, PR #480"
  confidence: 0.95
  severity: 5
  tags: ["quantization", "precision", "catastrophic"]
  notes: "int4 on standard architecture is fundamentally incompatible; this is not an implementation issue"
```

### Failure 5: Cache LM on FineWeb

```yaml
id: "FM-005"
technique: "Unigram Cache LM"
technique_category: "test_time"
outcome:
  value: +0.002
  unit: "BPB"
  baseline_value: 1.1248
  delta: +0.002
  direction: "negative"
mechanism:
  causal_chain: "Cache LM requires document coherence to accumulate useful signal. FineWeb documents are short (avg ~500 tokens) and diverse, preventing the cache from building meaningful context. The cache adds noise rather than signal."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: null
conditions:
  hardware_required: null
  scale_range: null
  training_duration: null
  constraint_type: null
  dataset_type: "FineWeb (short, diverse web text)"
  prerequisites: []
counterfactual:
  condition_for_success: "Works on long, homogeneous documents (Wikipedia, code repositories)"
  required_change: "Dataset must have documents long enough for cache to accumulate context (> 2000 tokens average)"
  expected_benefit_if_fixed: -0.005  # Cache LM on Wikipedia-scale data
relationships:
  related_failures: []
  conflicts_with: ["short_document_dataset", "high_diversity_dataset"]
  requires: []
metadata:
  source_pr: "PR #183"
  confidence: 0.90
  severity: 2
  tags: ["cache", "dataset_mismatch", "document_length"]
  notes: "Cache LM is not fundamentally flawed; this is a dataset-technique mismatch"
```

### Failure 6: Depth Recurrence + int6 Quantization

```yaml
id: "FM-006"
technique: "Depth Recurrence + int6 Quantization"
technique_category: "architecture"
outcome:
  value: +1.14
  unit: "BPB"
  baseline_value: 1.1248
  delta: +1.14
  direction: "negative"
mechanism:
  causal_chain: "Weight sharing across 3 recurrence cycles amplifies int6 quantization error ~900x. Each recurrence passes the quantization error forward, compounding across cycles. The error that would be minor in a standard model becomes catastrophic."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "quantization error amplification"
conditions:
  hardware_required: null
  scale_range: null
  training_duration: null
  constraint_type: "size_constrained"
  dataset_type: null
  prerequisites: ["depth_recurrence", "int6_quantization"]
counterfactual:
  condition_for_success: "Use int8 or higher precision when combining with depth recurrence"
  required_change: "Depth recurrence is incompatible with int6; requires int8+ precision"
  expected_benefit_if_fixed: -0.010  # Theoretical gain from depth recurrence with proper precision
relationships:
  related_failures: ["FM-004"]
  conflicts_with: ["int6_quantization", "int5_quantization"]
  requires: ["int8_or_higher"]
metadata:
  source_pr: "PR #363"
  confidence: 0.95
  severity: 5
  tags: ["quantization", "architecture", "error_amplification", "catastrophic"]
  notes: "Depth recurrence without quantization works (~1.177 BPB); the failure is specific to the int6 combination"
```

### Failure 7: XSA + TTT Combined

```yaml
id: "FM-007"
technique: "Exclusive Self Attention (XSA) + Test-Time Training (TTT)"
technique_category: "architecture"
outcome:
  value: -0.016
  unit: "BPB"
  baseline_value: 1.1248
  delta: -0.016
  direction: "negative"
mechanism:
  causal_chain: "XSA and TTT both target local context modeling. XSA removes self-value bias to force the model to attend to other tokens. TTT fine-tunes on local context patterns. Combined, they create redundancy where each technique's benefit is cancelled by the other's mechanism."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "mechanism redundancy"
conditions:
  hardware_required: null
  scale_range: null
  training_duration: null
  constraint_type: null
  dataset_type: null
  prerequisites: ["XSA", "TTT"]
counterfactual:
  condition_for_success: "Use XSA OR TTT, but not both in same stack"
  required_change: "Choose one: XSA provides architectural improvement; TTT provides test-time adaptation. They conflict."
  expected_benefit_if_fixed: null
relationships:
  related_failures: []
  conflicts_with: []
  requires: []
metadata:
  source_pr: "PR #303"
  confidence: 0.90
  severity: 3
  tags: ["redundancy", "mechanism_conflict", "local_context"]
  notes: "Individual XSA is positive (-0.005 BPB); individual TTT is positive (-0.010 to -0.020 BPB); combined is negative"
```

### Failure 8: MoE Routing at Small Scale

```yaml
id: "FM-008"
technique: "Mixture of Experts (MoE) Routing"
technique_category: "architecture"
outcome:
  value: +0.016
  unit: "BPB"
  baseline_value: 1.1248
  delta: +0.016
  direction: "negative"
mechanism:
  causal_chain: "MoE soft-routing with 2 experts adds 12ms/step overhead. At small model scale (< 30M params), the expert routing capacity does not compensate for the overhead cost. The model lacks sufficient parameters for the routing mechanism to provide benefit."
  throughput_impact_ms_per_step: 12
  memory_impact_mb: null
  convergence_impact: "insufficient model capacity for routing benefit"
conditions:
  hardware_required: null
  scale_range: "< 30M params"
  training_duration: null
  constraint_type: "time_constrained"
  dataset_type: null
  prerequisites: []
counterfactual:
  condition_for_success: "Works at larger scale (> 500M params) where routing capacity benefit exceeds overhead"
  required_change: "Do not use MoE at small scale; or reduce overhead via hard routing"
  expected_benefit_if_fixed: null
relationships:
  related_failures: ["FM-001", "FM-002"]
  conflicts_with: ["small_model_scale", "time_constrained_training"]
  requires: []
metadata:
  source_pr: "PR #480"
  confidence: 0.85
  severity: 3
  tags: ["moe", "routing", "overhead", "scale"]
  notes: "MoE is not fundamentally flawed; the failure is specific to small-scale time-constrained regimes"
```

### Failure 9: Label Smoothing

```yaml
id: "FM-009"
technique: "Label Smoothing"
technique_category: "regularization"
outcome:
  value: 0.0
  unit: "BPB"
  baseline_value: 1.1248
  delta: 0.0
  direction: "neutral"
mechanism:
  causal_chain: "Label smoothing provides no benefit at this model scale and dataset. The model is already well-regularized by other techniques (weight decay, EMA). The additional regularization from label smoothing is redundant."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: null
conditions:
  hardware_required: null
  scale_range: "< 100M params"
  training_duration: null
  constraint_type: null
  dataset_type: null
  prerequisites: []
counterfactual:
  condition_for_success: "May provide benefit on larger models or with less regularization"
  required_change: "None needed; this is a neutral result, not a failure"
  expected_benefit_if_fixed: null
relationships:
  related_failures: ["FM-010"]
  conflicts_with: []
  requires: []
metadata:
  source_pr: "PR #375"
  confidence: 0.85
  severity: 1
  tags: ["regularization", "no_effect"]
  notes: "This is a neutral result, not a negative. Label smoothing neither helps nor hurts at this scale."
```

### Failure 10: L1 Regularization

```yaml
id: "FM-010"
technique: "L1 Regularization"
technique_category: "regularization"
outcome:
  value: 0.0
  unit: "BPB"
  baseline_value: 1.1248
  delta: 0.0
  direction: "neutral"
mechanism:
  causal_chain: "L1 regularization provides no benefit at this model scale. The sparsity-inducing effect conflicts with the dense weight distributions needed for quantization. L1 pushes weights toward zero, which can degrade quantization quality."
  throughput_impact_ms_per_step: null
  memory_impact_mb: null
  convergence_impact: "potential quantization degradation"
conditions:
  hardware_required: null
  scale_range: "< 100M params"
  training_duration: null
  constraint_type: "size_constrained"
  dataset_type: null
  prerequisites: []
counterfactual:
  condition_for_success: "May provide benefit in non-quantized models or with post-training quantization disabled"
  required_change: "Do not combine L1 with quantization-focused training"
  expected_benefit_if_fixed: null
relationships:
  related_failures: ["FM-009"]
  conflicts_with: ["quantization"]
  requires: []
metadata:
  source_pr: "PR #375"
  confidence: 0.85
  severity: 1
  tags: ["regularization", "no_effect", "sparsity"]
  notes: "This is a neutral result in the context of the competition; L1 may hurt if quantization is used"
```

---

## 7. Implementation Roadmap

### Phase 1: Manual Curation from Research Reports (Weeks 1-2)

**Goal**: Seed the FMKB with high-confidence entries from existing research documentation.

**Tasks**:
1. Implement FMKB database schema (SQLite)
2. Create structured entry format for failures
3. Extract all known failures from parameter_golf_research_report.md
4. Populate with the 10 documented failures above
5. Build basic query API (will_technique_fail, get_failure_mode)

**Deliverables**:
- Functional FMKB with 10+ seed entries
- Query API returning failure predictions
- Integration test with parameter golf stack planner

**Success criteria**:
- 100% of known parameter golf failures captured
- Query latency < 50ms for single-technique lookup
- All extracted entries have confidence >= 0.80

### Phase 2: LLM-Assisted Extraction from Experiment Logs (Weeks 3-5)

**Goal**: Automatically extract failure modes from new experiment results.

**Tasks**:
1. Build LLM extraction pipeline for PR/experiment descriptions
2. Implement automated log parsing for negative result detection
3. Add uncertainty scoring for extracted entries
4. Build user feedback capture UI
5. Implement confidence update based on user confirmation

**Deliverables**:
- Automated extraction from PR descriptions with > 80% precision
- User feedback workflow (report → validate → confirm)
- Confidence scoring engine

**Success criteria**:
- New experiment failures extracted and entered within 1 hour of PR
- User confirmation workflow reduces false positive rate by > 50%
- Confidence scores correlate with extraction quality

### Phase 3: Proactive Failure Prediction (Weeks 6-8)

**Goal**: Predict failures before technique adoption.

**Tasks**:
1. Build context-aware failure prediction (Query A)
2. Implement dependency graph integration
3. Add precondition inference from failure entries
4. Build conflict detection for technique pairs
5. Implement throughput budget calculator (Query C)

**Deliverables**:
- Full query interface (A, B, C, D, E)
- Dependency graph updates from FMKB entries
- Planning-stage failure warnings

**Success criteria**:
- Query A returns correct failure predictions for > 85% of known failure conditions
- Graph updates correctly infer prerequisites from failure mechanisms
- Throughput calculator matches measured overhead within 10%

### Phase 4: Knowledge Base Expansion (Ongoing)

**Goal**: Grow FMKB beyond parameter golf to general ML training knowledge.

**Tasks**:
1. Extract failures from external ML literature
2. Build community contribution workflow
3. Implement cross-techniqueKB linking (similar failures in different domains)
4. Add temporal awareness (failures that emerged with new hardware/datasets)

**Deliverables**:
- External literature extraction pipeline
- Community contribution portal
- Cross-domain failure linking

---

## Appendix: Implementation Notes

### Technology Stack
- **Database**: SQLite with FTS5 (simple, portable, sufficient for MVP)
- **Query Engine**: Python rules engine + LLM for complex condition matching
- **Integration**: REST API for QKayV core to query FMKB
- **Future**: Consider vector database for semantic similarity if knowledge base grows large

### Key Design Decisions

1. **Separate from dependency graph initially**: FMKB and dependency graph are closely linked but have different update frequencies. FMKB entries are added infrequently; dependency graph edges can be derived from FMKB. Keeping them separate simplifies implementation.

2. **Confidence-weighted queries**: Rather than binary "will fail / won't fail", queries return confidence-weighted risk scores. This handles partial failures and uncertain conditions gracefully.

3. **Mechanism-focused**: The most valuable part of a failure entry is the mechanism. This is what allows QKayV to generalize from "MTP failed" to "any high-overhead technique will fail in time-constrained settings."

### Out of Scope for MVP
- Visualization UI for failure mode relationships
- Automated technique suggestion based on FMKB
- Cross-experiment failure correlation analysis
- Integration with external ML experiment trackers (Weights & Biases, MLflow)
