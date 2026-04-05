# Technique Dependency Graph — Implementation Plan

**Document**: QKayV Feature Specification — Technique Dependency Graph
**Date**: 2026-03-24
**Status**: Planning
**Ground Truth Source**: `parameter_golf_research_report.md` (560+ PRs, 21 records, 11 issues)

---

## Overview

The Technique Dependency Graph (TDG) is a knowledge structure that captures the combinatorial structure of model training techniques — which techniques conflict, which are orthogonal, what constraints cascade through the system, and what must be true before a technique can be applied. The TDG is the backbone of QKayV's ability to suggest next experiments, flag dangerous combinations, and propagate constraints through a proposed stack.

---

## 1. Data Model

### 1.1 Core Entities

The graph has three primary entity types: **Techniques**, **Constraints**, and **Artifacts**.

#### Technique Node

A technique is any trainable, architectural, or evaluative modification that affects model quality or training efficiency.

```python
@dataclass
class Technique:
    id: str                              # canonical slug: "xsa", "mlp3x", "int6_qat"
    name: str                             # human-readable: "Exclusive Self Attention"
    category: TechniqueCategory           # enum: QUANTIZATION, ARCHITECTURE, OPTIMIZER, EVALUATION, DATA, TESTTIME
    parameters: dict[str, ParameterSpec]  # named parameters with ranges/defaults

    # --- Cost Model ---
    cost_model: CostModel                  # how this technique affects budget
    param_delta: int                       # net parameter count change (can be negative)
    throughput_cost_ms: float              # per-step overhead in milliseconds
    quant_interaction: QuantInteraction    # how technique interacts with quantization

    # --- Combinatorial Structure ---
    preconditions: list[Precondition]     # what must be true before applying
    conflicts_with: list[ConflictEdge]   # explicit hard conflicts
    requires_mutually: list[str]          # techniques that MUST be applied together
    is_stackable_with: list[str]          # explicit positive interactions (orthogonal)
    orthogonal_cluster: str | None        # cluster ID for mutually-compatible techniques

    # --- Validation State ---
    validation_status: ValidationStatus    # MULTI_SEED_VALIDATED, SINGLE_SEED, ANECDOTAL, FAILED
    evidence: list[EvidenceRef]          # PRs, records, papers citing this technique
    failure_modes: list[FailureMode]      # known failure conditions

    # --- Metadata ---
    discovered_in: str                     # PR or record where first validated
    transferability: Transferability       # HIGH, MEDIUM, LOW, CONTEXT_SPECIFIC
    notes: str                             # free-text caveats
```

**Concrete examples from the report:**

| Technique ID | category | param_delta | throughput_cost_ms | quant_interaction |
|---|---|---|---|---|
| `int6_qat` | QUANTIZATION | 0 | 0 | STE gradient; eliminates roundtrip gap |
| `mlp3x` | ARCHITECTURE | +4M | 0 | Requires int6 or int5 to fit budget |
| `xsa` | ARCHITECTURE | 0 | +2ms (efficient impl) | No direct quant interaction |
| `ttt_score_first` | TESTTIME | 0 | varies (eval overhead) | Compatible with int6 |
| `wd_0_04` | OPTIMIZER | 0 | 0 | Controls artifact compressibility |
| `depth_recurrence` | ARCHITECTURE | 0 | varies | AMPLIFIES int6 quant error 900x |
| `int4` | QUANTIZATION | -50% | 0 | Catastrophic gap (+0.065 BPB) |

#### Parameterized Techniques

Techniques with continuous parameters are represented as a base technique with a parameter space:

```python
@dataclass
class ParameterizedTechnique:
    base_id: str                    # "weight_decay"
    parameter_name: str             # "decay"
    default_value: float             # 0.01
    validated_range: tuple[float, float]  # (0.0, 0.1)
    effect_profile: dict[float, EffectPoint]  # value -> {bpb_delta, throughput_delta}
```

**Concrete examples:**

- `weight_decay`: WD=0.04 vs WD=0.01 produces measurably different artifact sizes (~1.5-2MB per 0.01 WD unit), not just quality differences. A single technique entry with `parameter_name="decay"` and `effect_profile` captures this.

- `int6_quant`: Per-row vs per-tensor clipping changes the effect profile. GPTQ-lite (5 percentile search) changes it further.

- `bigram_hash`: Bucket count 4096 vs 8192 vs 10240 vs 12288 has diminishing returns.

- `partial_rope`: Fraction of dimensions (25% = 16/64) is the key parameter; effect is non-linear.

### 1.2 Edge Types

```python
@dataclass
class DependencyEdge:
    source: str            # technique ID
    target: str            # technique ID
    edge_type: EdgeType
    strength: float        # 0.0-1.0, confidence in this relationship
    evidence: list[str]    # PR numbers, record IDs
    context_conditions: list[str]  # "requires: int6_quant", "only: 11_layers"

EdgeType = Enum("EdgeType", [
    "CONFLICTS",       # hard negative: never combine
    "REDUNDANT",       # targets overlap mechanisms; combined gain is less than sum
    "REQUIRES",        # target cannot be applied without source
    "ENABLES",         # source creates headroom for target (e.g., int6 -> mlp3x)
    "ORTHOGONAL",      # no interaction; can stack freely
    "CASCADE",         # source constrains target's valid parameter space
])
```

**Concrete edge examples from the report:**

```
Edge(type=ENABLES, source=int6_qat, target=mlp3x,
     strength=1.0, evidence=[PR70, Record#6, Record#9, Record#10],
     context_conditions=["budget: 16MB"])
     # int6 headroom (~4MB saving vs int8) funds the 3x MLP expansion

Edge(type=CONFLICTS, source=xsa, target=ttt_score_first,
     strength=1.0, evidence=[PR303],
     context_conditions=["any"])  # XSA + TTT both target local context; combined is -0.016 BPB

Edge(type=REDUNDANT, source=xsa, target=ttt_adapt_first,
     strength=0.9, evidence=[Issue402],
     context_conditions=["any"])
     # Adapt-first TTT is information leakage, but also mechanism overlap with XSA

Edge(type=CASCADE, source=wd_0_04, target=int6_quant,
     strength=0.8, evidence=[PR375, PR238],
     context_conditions=["zstd_compression"])
     # Higher WD produces weights that compress better under zstd; extends to quant context too

Edge(type=CONFLICTS, source=depth_recurrence, target=int6_qat,
     strength=1.0, evidence=[PR363],
     context_conditions=["any"])
     # Depth recurrence amplifies int6 quant error by 900x; int8 is the minimum viable precision

Edge(type=ORTHOGONAL, source=partial_rope, target=ln_scale,
     strength=1.0, evidence=[Record#14],
     context_conditions=["any"])
     # Partial RoPE + LN Scale are independently motivated and combine cleanly

Edge(type=ORTHOGONAL, source=xsa, target=ln_scale,
     strength=1.0, evidence=[Record#14],
     context_conditions=["any"])
     # Zero-parameter architectural tricks; no interaction

Edge(type=ENABLES, source=mixed_int5_mlp, target=mlp3x,
     strength=0.9, evidence=[PR544],
     context_conditions=["budget: 16MB"])
     # int5 MLP is more aggressive than int6, frees more headroom for wider MLP

Edge(type=CONFLICTS, source=int4, target=[xsa, smgargate, bigram_hash, ortho_init],
     strength=1.0, evidence=[PR367],
     context_conditions=["bitnet_ternary"])
     # Standard techniques break in ternary regime
```

### 1.3 Constraint Nodes

```python
@dataclass
class Constraint:
    id: str
    type: ConstraintType        # BUDGET_SIZE, THROUGHPUT, ACCURACY, HARDWARE, TRAINING_STEPS
    operator: str               # "<=", ">=", "==", "range"
    value: any                  # numeric value or tuple
    unit: str                   # "MB", "ms/step", "BPB", "layers", "epochs"

    # Propagation rules
    propagated_through: list[Edge]    # which edges this constraint cascades through
    derived_constraints: list[str]    # constraints that are implied by this one

@dataclass
class ConstraintPropagation:
    source_constraint: str
    path: list[str]             # technique IDs along the propagation path
    resulting_constraint: str   # derived constraint
    mechanism: str              # human-readable mechanism description
```

**Concrete constraint propagation example:**

```
Constraint(id="budget_16mb", type=BUDGET_SIZE, operator="<=", value=16, unit="MB")

Propagation path:
  budget_16mb
    -> ENABLES -> int6_quant [saves ~4MB vs int8 on 9-layer model]
    -> ENABLES -> mixed_int5_mlp [saves additional ~1-2MB]
    -> ENABLES -> mlp3x [funded by quant headroom]
    -> CASCADE -> wd_0_04 [higher WD makes weights compress better]
    -> CASCADE -> warmdown_extended [better quant-friendly weight distribution]

ConstraintPropagation(
    source_constraint="budget <= 16MB",
    path=["int6_quant", "mlp3x"],
    resulting_constraint="mlp3x requires int6 or int5 to fit",
    mechanism="int6 saves ~4MB over int8, which directly funds the 3x MLP expansion (+4M params)"
)
```

### 1.4 Evidence References

```python
@dataclass
class EvidenceRef:
    source_type: str           # "pr", "record", "issue", "paper", "experiment_log"
    source_id: str             # e.g., "PR#303", "Record#11", "PR#70"
    url: str | None
    bpb_delta: float | None    # measured effect on quality
    throughput_delta: float | None  # ms/step overhead
    conditions: list[str]      # "11_layers", "int6_quant", "H100_only"
    seeds: int                  # number of seeds validated
    is_positive: bool          # True for improvements, False for failures
```

### 1.5 Graph Storage

The graph should be stored as structured JSON with referential integrity:

```json
{
  "version": "1.0",
  "techniques": { ... },
  "edges": [ ... ],
  "constraints": { ... ],
  "propagation_rules": [ ... ],
  "orthogonal_clusters": [
    {
      "id": "zero_param_arch",
      "members": ["xsa", "partial_rope", "ln_scale"],
      "description": "Zero-parameter architectural tricks that combine freely"
    }
  ]
}
```

---

## 2. Knowledge Acquisition

### 2.1 Manual Extraction (Phase 1 MVP)

The initial graph is built from `parameter_golf_research_report.md` using structured extraction.

**Process:**
1. A researcher or LLM reviews each record and PR, extracting technique interactions into a standardized form
2. The graph is populated incrementally as the report is built
3. Each extraction includes the evidence references for provenance

**Extraction template for each PR/record:**

```
TECHNIQUE: <list of technique IDs introduced or used>
CATEGORY: <per technique>
BPB_DELTA: <measured improvement or degradation>
THROUGHPUT_COST: <ms/step if reported>
PARAM_COUNT: <delta if reported>
CONFLICTS: <technique IDs that this PR found to conflict>
ENABLES: <technique IDs that this PR found to enable>
CONDITIONS: <architecture constraints, hardware constraints, training constraints>
STATUS: <POSITIVE, NEGATIVE, MIXED, ANECDOTAL>
EVIDENCE: <PR number, record ID>
```

**Concrete extraction examples from the report:**

From PR #303 (XSA + TTT negative interaction):
```
TECHNIQUES: [xsa, ttt_score_first]
CONFLICTS: [xsa, ttt_score_first]  # bidirectional conflict discovered
BPB_DELTA: +0.016 (NEGATIVE — they target overlapping mechanisms)
CONDITIONS: [any]
STATUS: NEGATIVE
EVIDENCE: PR#303
```

From Record #6 (int6 + MLP3x foundation):
```
TECHNIQUES: [int6_per_row, mlp3x]
ENABLES: {int6_per_row -> mlp3x}  # int6 headroom funds wider MLP
BPB_DELTA: -0.030 combined
CONDITIONS: [budget_16mb, qat_required]
STATUS: POSITIVE
EVIDENCE: Record#6, PR#70
```

### 2.2 LLM-Assisted Extraction (Phase 2)

As experiment logs accumulate, LLM-assisted extraction becomes possible:

**Input**: Raw PR descriptions, experiment log snippets, slack discussions, meeting notes

**Process**:
```python
def extract_from_text(text: str, context: dict) -> list[ExtractedTechnique | ExtractedEdge]:
    """
    LLM prompt:

    You are analyzing a researcher's experiment notes about model training.
    Extract any technique interactions described. For each, identify:
    1. Technique(s) being discussed
    2. Effect (positive, negative, neutral, unclear)
    3. Magnitude if reported (BPB, params, ms/step)
    4. Conditions under which it was tested
    5. Conflict or synergy with other techniques mentioned
    6. Confidence level (high/medium/low)

    Format your response as a JSON array of extractions.
    If no technique interactions are described, return [].
    """

    # The LLM returns structured JSON
    # A validation layer checks for consistency with existing graph
    # Conflicts are flagged for human review
    # High-confidence extractions are auto-merged; low-confidence ones are queued
```

**Confidence thresholds:**
- HIGH (evidence from multiple independent sources, quantitative measurement): auto-merge
- MEDIUM (single source, quantitative): queue for review
- LOW (anecdotal, qualitative): store in separate "unverified" partition

### 2.3 Handling Uncertainty

The graph MUST represent uncertainty explicitly, not sweep it under the rug.

**Uncertainty representation:**

```python
@dataclass
class UncertaintyBand:
    bpb_low: float
    bpb_high: float
    throughput_low: float
    throughput_high: float
    confidence: float           # 0.0-1.0
    explanation: str             # "single seed", "different batch sizes", "H100 only"

# Example: XSA effect
Technique(
    id="xsa",
    bpb_effect=UncertaintyBand(-0.006, -0.004, 0, +2),
    # -0.006 to -0.004 BPB, 0 to +2ms/step depending on efficient vs naive implementation
)
```

**Context-dependent validity:**

Many techniques work in some contexts but not others:

```python
@dataclass
class ContextualValidity:
    technique_id: str
    valid_contexts: list[ContextSpec]   # contexts where it works
    invalid_contexts: list[ContextSpec] # contexts where it fails

ContextSpec = dict[str, str | list[str]]  # e.g., {"layers": ">=9", "quant": "int6_or_better"}

# Example: int5 MLP validity
ContextualValidity(
    technique_id="mixed_int5_mlp",
    valid_contexts=[
        {"training_steps": ">=15000", "model_size": ">=20M"},
        {"convergence": "well_trained"}
    ],
    invalid_contexts=[
        {"training_steps": "<10000", "model_size": "<20M"},
        {"note": " undertrained models have catastrophic +1.1 BPB gap per PR#238"}
    ]
)
```

### 2.4 Handling New Techniques

New techniques discovered during research are added with a lifecycle:

```python
@dataclass
class TechniqueLifecycle:
    NEW = "new"                    # just discovered, no validation
    SINGLE_SEED = "single_seed"    # validated on one seed
    MULTI_SEED = "multi_seed"      # validated across multiple seeds
    FAILED = "failed"             # found to be net negative
    CONTEXTUAL_FAILURE = "contextual_failure"  # fails in some contexts
    RETIRED = "retired"            # superseded by better technique
```

---

## 3. Query Interface

### 3.1 Core Query: What can be added to a stack?

Given a current technique stack, find valid techniques to add.

**Input**: Current stack `[t1, t2, ..., tn]`, optional budget constraints

**Output**: Ranked list of `(technique, reason, expected_delta)`

```python
def query_addable_techniques(
    current_stack: list[str],
    constraints: dict[str, Constraint] | None = None,
    max_results: int = 10
) -> list[TechniqueSuggestion]:
    """
    1. Find all techniques not already in stack
    2. Filter out hard conflicts with any stack member
    3. Filter out techniques whose preconditions are not satisfied
    4. Apply constraint propagation (e.g., budget -> int6 required -> mlp3x enabled)
    5. Score by expected BPB improvement per unit cost (throughput-adjusted)
    6. Return ranked suggestions with reasoning
    """

@dataclass
class TechniqueSuggestion:
    technique: str
    expected_bpb_delta: float
    expected_throughput_delta_ms: float
    throughput_adjusted_bpb: float  # the real figure of merit
    reason: str                      # why this is suggested
    confidence: float
    preconditions_met: list[str]
    preconditions_missing: list[str]
    conflicts_resolved: list[str]    # what was overridden or worked around
```

**Concrete example:**

```python
current_stack = ["int6_qat", "mlp3x", "wd_0_04", "ema"]
query_addable_techniques(current_stack)

# Expected output (from report data):
# 1. partial_rope: -0.0023 BPB, 0ms, orthogonal (LN Scale already present)
# 2. xsa: -0.005 BPB, +2ms, orthogonal with current stack
# 3. ln_scale: -0.001 BPB, 0ms, orthogonal
# 4. bigram_hash: -0.01 BPB, +1ms, orthogonal
# 5. gptq_lite: -0.0006 BPB, +5sec eval time, compatible with int6

# Techniques filtered out:
# - ttt_score_first: CONFLICTS with xsa (redundant mechanisms)
# - depth_recurrence: CONFLICTS with int6 (900x quant error amplification)
# - int4: fails int6 precondition not met
```

### 3.2 Core Query: What conflicts with a proposed technique?

```python
def query_conflicts(
    proposed_technique: str,
    current_stack: list[str],
    context: dict | None = None
) -> list[ConflictReport]:
    """
    Find all conflicts between proposed technique and current stack.
    Returns both hard conflicts and soft (redundant) interactions.
    """

@dataclass
class ConflictReport:
    conflicting_technique: str
    edge_type: EdgeType
    strength: float
    evidence: list[str]
    mechanism: str              # why this conflicts
    suggestion: str | None      # how to resolve if possible
```

**Concrete examples from the report:**

```python
query_conflicts("ttt_score_first", ["xsa", "ema"])

# Returns:
# ConflictReport(
#     conflicting_technique="xsa",
#     edge_type=REDUNDANT,
#     strength=1.0,
#     evidence=["PR#303"],
#     mechanism="XSA and TTT both target local context modeling; combined gain is less than sum",
#     suggestion="Remove XSA if using TTT, or use TTT without XSA"
# )
```

```python
query_conflicts("depth_recurrence", ["int6_qat"])

# Returns:
# ConflictReport(
#     conflicting_technique="int6_qat",
#     edge_type=CONFLICTS,
#     strength=1.0,
#     evidence=["PR#363"],
#     mechanism="Depth recurrence amplifies int6 quant error by 900x; gap goes from +0.001 to +1.14 BPB",
#     suggestion="Use int8 or higher if using depth recurrence"
# )
```

### 3.3 Core Query: Find orthogonal technique clusters

```python
def query_orthogonal_clusters(
    current_stack: list[str],
    min_cluster_size: int = 2
) -> list[OrthogonalCluster]:
    """
    Find groups of techniques that can be freely stacked with current stack.
    These are candidates for joint optimization or ablation studies.
    """

@dataclass
class OrthogonalCluster:
    cluster_id: str
    techniques: list[str]
    combined_expected_bpb: float
    combined_throughput_cost: float
    mutual_interactions: list[str]  # "no known interactions between any pair"
```

**Concrete example:**

```python
query_orthogonal_clusters(["int6_qat", "mlp3x"])

# Returns cluster:
# OrthogonalCluster(
#     cluster_id="zero_param_arch",
#     techniques=["xsa", "partial_rope", "ln_scale"],
#     combined_expected_bpb=-0.008,  # -0.005 + -0.0023 + -0.001
#     combined_throughput_cost=+2,  # xsa adds ~2ms; others are zero
#     mutual_interactions=["all pairwise orthogonal per Record#14 and Record#11"]
# )

# This cluster is the foundation stack used in Record#14:
#   11L + Partial RoPE + LN Scale + EMA + XSA4
#   Combined: -0.008 BPB for zero added parameters and minimal overhead
```

### 3.4 Core Query: Constraint propagation

```python
def propagate_constraints(
    initial_constraints: list[Constraint],
    direction: str = "downstream"  # or "upstream"
) -> PropagationResult:
    """
    Given a set of constraints (e.g., budget <= 16MB), trace through the graph
    to find all implied constraints and enabled/forbidden techniques.
    """

@dataclass
class PropagationResult:
    original_constraints: list[Constraint]
    implied_constraints: list[Constraint]
    enabled_techniques: list[str]   # techniques now viable
    disabled_techniques: list[str]  # techniques now infeasible
    paths: list[ConstraintPropagation]
```

**Concrete example:**

```python
propagate_constraints([Constraint(type=BUDGET_SIZE, operator="<=", value=16, unit="MB")])

# Returns:
# PropagationResult(
#     original_constraints=[budget <= 16MB],
#     implied_constraints=[
#         "int6_or_better required",
#         "mlp3x only if int6_or_better",
#         "wd_0_04 recommended for compressibility"
#     ],
#     enabled_techniques=["int6_qat", "int6_per_row", "mixed_int5_mlp", "mlp3x"],
#     disabled_techniques=["int4", "depth_recurrence"],
#     paths=[
#         ConstraintPropagation(
#             source_constraint="budget <= 16MB",
#             path=["int6_qat"],
#             resulting_constraint="int6 required for this budget",
#             mechanism="int6 saves ~4MB over int8 baseline, enabling 3x MLP within budget"
#         ),
#         ConstraintPropagation(
#             source_constraint="budget <= 16MB",
#             path=["int6_qat", "mlp3x"],
#             resulting_constraint="mlp3x requires int6 headroom",
#             mechanism="MLP 3x adds ~4M params; int6 funds this from int8 baseline savings"
#         )
#     ]
# )
```

### 3.5 Advanced Query: Ablation plan generation

```python
def generate_ablation_plan(
    full_stack: list[str],
    control_stack: list[str] | None = None  # baseline to compare against
) -> AblationPlan:
    """
    Given a full technique stack, generate a systematic ablation plan
    that tests each technique's contribution in isolation and in subgroups.
    """

@dataclass
class AblationPlan:
    steps: list[AblationStep]
    estimated_runs: int
    estimated_total_time: str
    priority_order: list[str]  # techniques to ablate first by impact

@dataclass
class AblationStep:
    stack: list[str]
    technique_removed: str | None  # None means baseline
    technique_added: str | None
    expected_delta: float
    purpose: str  # "isolate X's effect", "test interaction with Y"
```

**Concrete example:**

```python
full_stack = ["int6_qat", "mlp3x", "xsa", "partial_rope", "ln_scale", "wd_0_04",
              "ema", "bigram_hash", "gptq_lite", "sliding_window_eval"]

generate_ablation_plan(full_stack)

# Ablation plan (prioritized by reported impact):
# Step 1: Remove bigram_hash -> expected +0.010 BPB (largest single contribution after foundation)
# Step 2: Remove mlp3x -> expected +0.020 BPB (largest architectural gain)
# Step 3: Remove xsa -> expected +0.005 BPB
# Step 4: Remove partial_rope -> expected +0.0023 BPB
# Step 5: Remove ln_scale -> expected +0.001 BPB
# Step 6: Remove wd_0_04 -> expected size increase (affects quant)
# Step 7: Remove int6_qat -> expect catastrophic +0.048 BPB (foundation)
# Step 8: Test xsa + ttt_score_first -> expected +0.016 BPB (PR#303 found this is negative)
```

### 3.6 Advanced Query: Joint optimization suggestions

```python
def suggest_joint_optimizations(
    current_stack: list[str],
    budget_type: str,  # "size", "time", "quality"
    budget_value: float,
    unit: str
) -> list[JointOptimization]:
    """
    Find groups of techniques that should be optimized together because
    they share a constraint or enable each other's headroom.
    """
```

**Concrete example:**

```python
suggest_joint_optimizations(["base_architecture"], budget_type="size", budget_value=16, unit="MB")

# Returns:
# JointOptimization(
#     group=["int6_qat", "mlp3x"],
#     mechanism="int6 saves ~4MB vs int8, mlp3x costs ~4MB; net zero size, +0.02 BPB",
#     expected_bpb_delta=-0.020,
#     expected_size_delta=0,  # neutral
#     confidence=1.0,
#     evidence=["Record#6", "PR#70"]
# )
#
# JointOptimization(
#     group=["int6_qat", "wd_0_04"],
#     mechanism="Higher WD produces more compressible weights; int6 compress ratio improves",
#     expected_bpb_delta=-0.002,  # via better quant
#     expected_size_delta=-0.5,  # MB estimate
#     confidence=0.8,
#     evidence=["PR#375"]
# )
```

---

## 4. Integration with Research Loop

### 4.1 Decision-Making Integration

The TDG feeds into QKayV's core decision-making at three points:

**Point 1: Suggesting next experiments**

When a researcher asks "what should I try next?", QKayV queries the graph:

```python
def get_next_experiment_suggestions(
    current_stack: list[str],
    constraints: dict,
    n: int = 5
) -> list[ExperimentSuggestion]:
    """
    1. Find all valid addable techniques (Section 3.1)
    2. For each, calculate throughput-adjusted expected improvement
    3. Filter by constraints (size budget, time budget, hardware)
    4. Rank by expected improvement per unit cost
    5. Return top N with full reasoning
    """
```

The output is not just "try X" but "try X because Y, but beware of Z":

```
SUGGESTION 1: Add BigramHash (4096-10240 buckets)
  Expected: -0.010 BPB
  Throughput cost: +1ms/step
  Throughput-adjusted: -0.010 BPB (cost is negligible)
  Confidence: HIGH (50+ independent reproductions)
  Reasoning: Your current stack has no n-gram feature; BigramHash recovers
             statistics lost by small vocab (1024). Diminishing returns above
             8192 buckets; start at 8192.
  Conflicts: None with current stack
  Preconditions: None
```

**Point 2: Pre-flight conflict checking**

Before a user submits an experiment, QKayV automatically checks:

```python
def preflight_check(stack: list[str], technique_to_add: str) -> PreflightReport:
    """
    Returns warnings and errors before an experiment runs.
    Errors block the run; warnings can be overridden.
    """

@dataclass
class PreflightReport:
    errors: list[PreflightError]   # must fix before running
    warnings: list[PreflightWarning]  # can override
    info: list[str]

PreflightError(
    code="HARD_CONFLICT",
    message="depth_recurrence + int6_qat: PR#363 found 900x quant error amplification",
    conflicting_pair=["depth_recurrence", "int6_qat"],
    severity="ERROR"
)

PreflightWarning(
    code="THROUGHPUT_COST",
    message="xsa + ttt_score_first: PR#303 found redundant mechanisms; combined -0.016 BPB",
    conflicting_pair=["xsa", "ttt_score_first"],
    severity="WARNING",
    can_override=True
)
```

**Point 3: Constraint-aware architecture search**

When searching for architecture changes, QKayV propagates constraints through the graph:

```
User: "I want to fit a 30M parameter model in 16MB"
Constraint propagation:
  30M params + 16MB -> int6 required
  int6 -> QAT or small quant gap required
  int6 -> wd_0_04 recommended (improves compressibility)
  int6 headroom (~4MB vs int8) -> mlp3x is viable

User: "What if I use depth recurrence?"
Conflict check: depth_recurrence conflicts with int6 (PR#363)
Suggestion: Use int8 if depth recurrence is required, or drop depth recurrence
```

### 4.2 Ablation Plan Integration

When a user asks "which techniques are actually contributing?", QKayV generates an ablation plan from the graph:

```python
def generate_ablation_plan_for_stack(stack: list[str]) -> AblationPlan:
    """
    Uses the graph's edge structure to determine the minimal set of
    ablations needed to isolate each technique's contribution.
    """
```

The graph structure enables smarter ablation: techniques in orthogonal clusters can be ablated together or separately; techniques with strong interactions require full grid ablation.

### 4.3 Joint Optimization Integration

When a user asks "can I optimize size and quality together?", QKayV finds technique groups that have complementary cost models:

```
User: "I want to improve quality without increasing model size"
Joint optimization query:
  Group 1: int6_qat + mlp3x
    - int6 saves ~4MB
    - mlp3x costs ~4MB but +0.02 BPB
    - Net: same size, better quality

  Group 2: wd_0_04 + int6_qat
    - wd_0_04 improves compressibility by ~0.5-1MB
    - int6 is more sensitive to weight distribution
    - Net: same or smaller size, better quality via quant

  Group 3: xsa + partial_rope + ln_scale
    - All zero parameter cost
    - All orthogonal
    - Combined: -0.008 BPB
```

---

## 5. Maintenance

### 5.1 Updating the Graph When New Failure Modes Are Discovered

Failure modes are first-class citizens in the graph, not afterthoughts.

**Update protocol:**

```python
def report_failure_mode(
    technique: str,
    failure_evidence: EvidenceRef,
    context: ContextSpec,
    mechanism: str,
    severity: str  # "hard_block", "degradation", "marginal"
) -> GraphUpdate:
    """
    1. Add failure mode to technique.failure_modes
    2. Create or update CONFLICTS edges with affected techniques
    3. Propagate new constraints through graph
    4. Flag all suggestions that included this technique in affected contexts
    5. Notify relevant users who used this technique
    """
```

**Concrete example: PR #363 discovers depth_recurrence + int6 failure**

```python
report_failure_mode(
    technique="depth_recurrence",
    failure_evidence=EvidenceRef(source_type="pr", source_id="PR#363"),
    context={"quantization": "int6", "layers": "any"},
    mechanism="Weight-sharing across 3 recurrence cycles amplifies int6 gap from +0.001 to +1.14 BPB",
    severity="hard_block"
)

# Resulting graph updates:
# - depth_recurrence.failure_modes += ["int6 quantization amplifies error 900x"]
# - New CONFLICTS edge: depth_recurrence <-> int6_qat (strength=1.0)
# - int6_qat.disabled_contexts += ["depth_recurrence"]
# - All suggestions containing depth_recurrence in int6 context get flagged
```

### 5.2 Handling Contradictory Findings

Contradictions are stored with full provenance and context.

```python
@dataclass
class Contradiction:
    finding_a: EvidenceRef
    finding_b: EvidenceRef
    techniques: list[str]
    context_diff: str             # what different conditions caused the contradiction
    proposed_resolution: str | None
    status: str                   # "unresolved", "context_dependent", "genuine_contradiction"

# Example: int5 MLP appears in both positive and negative contexts

Contradiction(
    finding_a=EvidenceRef(source_type="pr", source_id="PR#544", bpB_delta=-0.003),
    finding_b=EvidenceRef(source_type="pr", source_id="PR#238", bpB_delta=+1.1),
    techniques=["mixed_int5_mlp"],
    context_diff="PR#544: well-trained 33.6M params, 15000+ steps; PR#238: undertrained, shorter training",
    proposed_resolution="int5 MLP viability is strongly training-duration dependent",
    status="context_dependent"
)
```

**Resolution rules:**
- If contradiction is context-dependent: split the technique's validity into multiple ContextualValidity entries
- If genuine contradiction: flag for human review, store both findings with confidence reduced
- If one finding is higher-confidence (more seeds, more reproductions): deweight or supersede the other

### 5.3 Graph Versioning and Rollback

Every update to the graph creates an immutable version:

```python
@dataclass
class GraphVersion:
    version_id: str
    timestamp: datetime
    changes: list[GraphDelta]
    author: str  # human or automated
    reason: str
    rolled_back: bool = False

@dataclass
class GraphDelta:
    entity_type: str  # "technique", "edge", "constraint"
    entity_id: str
    operation: str     # "add", "remove", "modify"
    before: any
    after: any
```

This enables rollback if a later update introduces errors, and provides full audit trail for research credibility.

---

## 6. Implementation Roadmap

### Phase 1: MVP — Manual Curation from Research Report

**Timeline**: 1-2 weeks
**Goal**: A working graph with manually extracted data from `parameter_golf_research_report.md`

**Deliverables:**
1. Core data model (Technique, Edge, Constraint classes)
2. JSON storage with ~40 technique nodes extracted from the report
3. ~60 edge instances covering all documented conflicts/enables/orthogonalities
4. Constraint propagation for the `budget -> int6 -> mlp3x` cascade
5. Query interface: `query_addable_techniques()`, `query_conflicts()`, `query_orthogonal_clusters()`
6. Integration: pre-flight conflict checking before experiment submission

**Concrete Phase 1 data (from report):**

Techniques to encode (from the report's technique taxonomy, filtering to VALIDATED entries):

| Category | Count | Examples |
|---|---|---|
| QUANTIZATION | 12 | int6_per_row, int6_qat, int5_mlp, gptq_lite, fp16_embed |
| ARCHITECTURE | 15 | mlp3x, xsa, partial_rope, ln_scale, value_residual, bigram_hash |
| OPTIMIZER | 6 | muon, ema, swa, wd_0_04, warmdown |
| EVALUATION | 2 | sliding_window_eval, stride_64 |
| TESTTIME | 3 | ttt_score_first, lora_ttt |
| DATA | 2 | seq2048, seq4096 |

Edges to encode (all documented in the report):
- 8 CONFLICTS edges (xsa<>ttt, depth_recurrence<>int6, int4<>standard_techniques, etc.)
- 5 ENABLES edges (int6<>mlp3x, int5<>mlp3x, wd<>compressibility, etc.)
- 3 ORTHOGONAL edges (partial_rope<>ln_scale, xsa<>ln_scale, xsa<>partial_rope)
- 4 CASCADE edges (budget->int6, wd->quant, warmdown->quant, etc.)

**What this enables:**
- "Given my current stack, what can I add?" query
- Pre-flight checking for known conflicts
- Constraint propagation for budget decisions
- First version of ablation plan generation

**What this does NOT enable:**
- Automated extraction from new PRs/experiments
- Real-time learning from user research sessions
- Uncertainty quantification beyond evidence counting

### Phase 2: Automated Extraction from Experiment Logs

**Timeline**: 2-4 weeks
**Goal**: LLM-assisted extraction from raw experiment outputs, PR descriptions, and research notes

**Deliverables:**
1. LLM extraction pipeline: unstructured text -> structured Technique/Edge
2. Validation layer: consistency checking against existing graph
3. Experiment log parser: captures techniques, metrics, conditions automatically
4. Uncertainty quantification: confidence bands on all effect estimates
5. Contradiction detection and resolution workflow
6. Graph versioning with rollback

**New capabilities:**
- New techniques discovered during research sessions are automatically added to the graph
- Effect sizes are updated as more data accumulates
- Conflicts are flagged for human review when detected (not auto-merged)
- "Unverified" partition for low-confidence findings

**Integration with QKayV logging:**
- When QKayV logs an experiment, the techniques and results are automatically extracted
- If a technique interaction is novel, it is queued for review
- If a technique interaction confirms existing graph entries, confidence increases

### Phase 3: Real-Time Learning from Research Sessions

**Timeline**: 4-8 weeks
**Goal**: The graph learns continuously from user interactions and experiment outcomes

**Deliverables:**
1. Online learning: graph updates as experiments complete (with appropriate lag for validation)
2. User feedback integration: researchers can confirm/reject suggestions, providing signal
3. Transfer learning: graph structures from parameter golf transfer to new domains
4. Causal inference: distinguish correlation from causation in technique interactions
5. Bayesian uncertainty: every effect estimate has a posterior distribution

**New capabilities:**
- "This combination worked well in my domain — does it change the graph?"
- Proactive suggestion: "Your recent experiments suggest X might work better than Y in your setup"
- Cross-domain transfer: "Techniques that work in LM training on this dataset also worked in these related tasks"

**Key challenge:** Avoiding feedback loops where the graph overfits to recent experiments at the expense of validated general knowledge. Mitigation: all online updates are treated as lower-confidence until validated by multiple independent users/experiments.

---

## Appendix: Concrete Data Structures Reference

### Full Technique Node Example (int6_qat)

```python
Technique(
    id="int6_qat",
    name="Int6 Quantization via Straight-Through Estimation",
    category=QUANTIZATION,
    parameters={},
    cost_model=CostModel(
        size_savings_mb=4.0,       # vs int8 on 9-layer model
        throughput_cost_ms=0,
        param_count_delta=0
    ),
    param_delta=0,
    throughput_cost_ms=0,
    quant_interaction=QuantInteraction(
        type="qat",
        mechanism="STE gradient passes through quantization roundtrip",
        roundtrip_gap_eliminated=True
    ),
    preconditions=[
        Precondition(
            type="TRAINING_METHOD",
            condition="QAT training or minimal quant gap",
            evidence=["Record#7", "PR#70"],
            is_required=True
        )
    ],
    conflicts_with=[
        ConflictEdge(
            target="depth_recurrence",
            reason="amplifies quant error 900x",
            evidence=["PR#363"],
            strength=1.0
        )
    ],
    is_stackable_with=[
        "mlp3x", "fp16_embed", "xsa", "partial_rope", "ln_scale",
        "wd_0_04", "ema", "sw_a", "bigram_hash", "gptq_lite"
    ],
    orthogonal_cluster=None,  # it's in the quantization cluster
    validation_status=ValidationStatus.MULTI_SEED,
    evidence=[
        EvidenceRef(source_type="pr", source_id="PR#70", bpB_delta=-0.030, is_positive=True),
        EvidenceRef(source_type="record", source_id="Record#6", is_positive=True),
        EvidenceRef(source_type="record", source_id="Record#7", is_positive=True),
    ],
    failure_modes=[
        FailureMode(
            condition="without QAT",
            effect="+0.048 BPB roundtrip gap",
            severity="degradation",
            evidence=["Record#6"]
        )
    ],
    discovered_in="PR#70",
    transferability=Transferability.HIGH,
    notes="STE QAT eliminates roundtrip gap. torch.compile dead code trap: late QAT flag can be constant-folded away (PR#315)"
)
```

### Full Edge Example (int6 -> mlp3x enables)

```python
DependencyEdge(
    source="int6_qat",
    target="mlp3x",
    edge_type=EdgeType.ENABLES,
    strength=1.0,
    evidence=["PR#70", "Record#6", "Record#9", "Record#10"],
    context_conditions=["budget: 16MB", "qat_required"],
    mechanism="int6 saves ~4MB over int8 baseline on 9-layer model; mlp3x adds ~4M params (~4MB at int6); net size change ~0",
    derived_constraints=[
        Constraint(
            id="mlp3x_requires_int6",
            type="BUDGET",
            operator="->",
            value="int6_or_better",
            unit="quantization_precision",
            implied_by=["int6_qat"]
        )
    ]
)
```

---

## Appendix: Report-Sourced Technique Interaction Matrix

| Technique A | Technique B | Relationship | Evidence | bpB Delta (combined) |
|---|---|---|---|---|
| int6_qat | mlp3x | ENABLES | PR#70 | -0.030 |
| xsa | ttt_score_first | CONFLICTS/REDUNDANT | PR#303 | +0.016 (negative) |
| depth_recurrence | int6_qat | CONFLICTS | PR#363 | +1.14 BPB gap |
| int4 | [xsa, smgargate, bigram_hash] | CONFLICTS | PR#367 | training plateaus |
| partial_rope | ln_scale | ORTHOGONAL | Record#14 | -0.0023 combined |
| xsa | ln_scale | ORTHOGONAL | Record#14 | no interaction |
| xsa | partial_rope | ORTHOGONAL | Record#14 | no interaction |
| wd_0_04 | int6_qat | CASCADE (improves) | PR#375 | better compression |
| ema | swa | REDUNDANT | PR#287 | ema > swa by 0.003 |
| int5_mlp | mlp3x | ENABLES | PR#544 | -0.003 (int5), enables wider |
| bigram_hash | mlp3x | ORTHOGONAL | Record#10 | -0.010 combined |
| gptq_lite | int6_qat | ORTHOGONAL | PR#379 | -0.0006 additive |

---

*Document generated from analysis of `parameter_golf_research_report.md` covering github.com/openai/parameter-golf through 560+ PRs, 21 record entries, 11 issues.*
