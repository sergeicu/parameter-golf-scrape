# Technique Dependency Graph — Implementation Plan v2

**Document**: QKayV Feature Specification — Technique Dependency Graph v2
**Date**: 2026-03-24
**Status**: Planning (v2 — addresses critique)
**Supersedes**: plan_dependency_graph.md v1
**Ground Truth Source**: `parameter_golf_research_report.md` (560+ PRs, 21 records, 11 issues)
**Critique Source**: `critique_dependency_graph.md`

---

## Overview

The Technique Dependency Graph (TDG) v2 is a typed, context-aware knowledge structure that captures the combinatorial structure of model training techniques. This revision addresses six critical gaps identified in the critique:

1. String-based `context_conditions` replaced with a typed `TechniqueContext` schema
2. Edge types separated to capture distinct mechanisms (budget reallocation vs. qualitative enablement; hard incompatibility vs. soft redundancy)
3. Partial/conditional validity (int5 MLP pattern) as a first-class data model feature
4. Constraint propagation with explicit traversal algorithm
5. QKayV integration defined explicitly with concrete integration points
6. Phase 1 scoped to a single validated technique stack with acceptance criteria

---

## 1. Data Model

### 1.1 Typed Context Schema — `TechniqueContext`

Every context condition is a typed field. No string-matching anywhere in the system.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

class QuantPrecision(Enum):
    INT4 = "int4"
    INT5 = "int5"
    INT5_MLP = "int5_mlp"     # int5 applied to MLP layers only
    INT6 = "int6"
    INT6_PER_ROW = "int6_per_row"
    INT8 = "int8"
    FP16 = "fp16"
    FP16_EMBED = "fp16_embed"

class ConvergenceState(Enum):
    UNTRAINED = "untrained"
    UNDERTRAINED = "undertrained"   # < 10000 steps (context-dependent)
    WELL_TRAINED = "well_trained"    # >= 15000 steps (context-dependent)
    CONVERGED = "converged"

class HardwareTarget(Enum):
    H100 = "h100"
    A100 = "a100"
    V100 = "v100"
    CPU = "cpu"
    ANY = "any"

class CompressionAlgorithm(Enum):
    ZSTD = "zstd"
    HUFFMAN = "huffman"
    NONE = "none"

@dataclass
class LayerRange:
    """Inclusive range of valid layer counts."""
    min_inclusive: int | None = None
    max_inclusive: int | None = None
    exact: int | None = None  # shorthand for min=max=exact

    def contains(self, n: int) -> bool:
        if self.exact is not None:
            return n == self.exact
        if self.min_inclusive is not None and n < self.min_inclusive:
            return False
        if self.max_inclusive is not None and n > self.max_inclusive:
            return False
        return True

@dataclass
class ModelSizeRange:
    """Inclusive range of valid model sizes in millions of parameters."""
    min_mparams: float | None = None
    max_mparams: float | None = None
    exact_mparams: float | None = None

    def contains(self, mparams: float) -> bool:
        if self.exact_mparams is not None:
            return abs(mparams - self.exact_mparams) < 0.1
        if self.min_mparams is not None and mparams < self.min_mparams:
            return False
        if self.max_mparams is not None and mparams > self.max_mparams:
            return False
        return True

@dataclass
class TrainingStepRange:
    """Inclusive range of valid training steps."""
    min_steps: int | None = None
    max_steps: int | None = None

    def contains(self, steps: int) -> bool:
        if self.min_steps is not None and steps < self.min_steps:
            return False
        if self.max_steps is not None and steps > self.max_steps:
            return False
        return True

@dataclass
class BudgetRange:
    """Size budget range in MB."""
    min_mb: float | None = None
    max_mb: float | None = None
    exact_mb: float | None = None

    def contains(self, mb: float) -> bool:
        if self.exact_mb is not None:
            return abs(mb - self.exact_mb) < 0.1
        if self.min_mb is not None and mb < self.min_mb:
            return False
        if self.max_mb is not None and mb > self.max_mb:
            return False
        return True

@dataclass
class TechniqueContext:
    """
    Typed context specification. All fields are optional (None = not constrained).
    A technique is valid in a given context if ALL non-None fields match.

    Example: int5 MLP validity from PR#544 vs PR#238:
      - PR#544 (valid):   layers=exact(11), model_size=exact(33.6M), convergence=WELL_TRAINED, steps=>=15000
      - PR#238 (invalid): layers=exact(11), model_size=<20M, convergence=UNDERTRAINED, steps=<10000
    """
    # Architecture constraints
    layers: LayerRange | None = None
    model_size_mparams: ModelSizeRange | None = None

    # Quantization constraints
    min_quant_precision: QuantPrecision | None = None   # "must use int6 or better"
    quant_precisions: list[QuantPrecision] | None = None  # "only int6 and int8 tested"

    # Training stage constraints
    convergence_state: ConvergenceState | None = None
    training_steps: TrainingStepRange | None = None

    # Hardware constraints
    hardware: list[HardwareTarget] | None = None

    # Compression constraints
    compression: CompressionAlgorithm | None = None

    # Budget constraints
    budget_mb: BudgetRange | None = None

    # Model architecture family
    architecture: str | None = None  # e.g., "decoder-only", "encoder-decoder"

    # Raw tag fallbacks for extensibility (only used when no typed field applies)
    # These are DEPRECATED — prefer adding a typed field
    raw_tags: list[str] = field(default_factory=list)

    def matches(self, other: "TechniqueContext") -> bool:
        """
        Returns True if self's non-None constraints are ALL satisfied by other's non-None constraints.
        self is a constraint specification; other is a query context.
        """
        if self.layers is not None and other.layers is not None:
            if not self.layers.contains(other.layers.exact or other.layers.min_inclusive or 0):
                return False
        if self.model_size_mparams is not None and other.model_size_mparams is not None:
            if not self.model_size_mparams.contains(other.model_size_mparams.exact_mparams or other.model_size_mparams.min_mparams or 0):
                return False
        if self.min_quant_precision is not None and other.min_quant_precision is not None:
            if not _quant_precision_satisfies(other.min_quant_precision, self.min_quant_precision):
                return False
        if self.quant_precisions is not None and other.quant_precisions is not None:
            if not any(p in self.quant_precisions for p in other.quant_precisions):
                return False
        if self.convergence_state is not None and other.convergence_state is not None:
            if self.convergence_state != other.convergence_state:
                return False
        if self.training_steps is not None and other.training_steps is not None:
            if not self.training_steps.contains(other.training_steps.min_steps or 0):
                return False
        if self.hardware is not None and other.hardware is not None:
            if not any(h in self.hardware for h in other.hardware):
                return False
        if self.compression is not None and other.compression is not None:
            if self.compression != other.compression:
                return False
        if self.budget_mb is not None and other.budget_mb is not None:
            if not self.budget_mb.contains(other.budget_mb.exact_mb or other.budget_mb.max_mb or float('inf')):
                return False
        return True

def _quant_precision_satisfies(actual: QuantPrecision, required: QuantPrecision) -> bool:
    """INT6 satisfies INT5, INT4; INT8 satisfies all."""
    hierarchy = {QuantPrecision.INT4: 4, QuantPrecision.INT5: 5,
                  QuantPrecision.INT5_MLP: 5, QuantPrecision.INT6: 6,
                  QuantPrecision.INT6_PER_ROW: 6, QuantPrecision.INT8: 8,
                  QuantPrecision.FP16: 16, QuantPrecision.FP16_EMBED: 16}
    return hierarchy.get(actual, 0) >= hierarchy.get(required, 0)
```

**Concrete examples:**

```python
# int5 MLP valid context (PR#544: 33.6M params, 15000+ steps, well-trained)
VALID_int5_mlp = TechniqueContext(
    layers=LayerRange(exact=11),
    model_size_mparams=ModelSizeRange(exact_mparams=33.6),
    convergence_state=ConvergenceState.WELL_TRAINED,
    training_steps=TrainingStepRange(min_steps=15000),
    min_quant_precision=QuantPrecision.INT5_MLP,
)

# int5 MLP invalid context (PR#238: undertrained, smaller model)
INVALID_int5_mlp = TechniqueContext(
    layers=LayerRange(exact=11),
    model_size_mparams=ModelSizeRange(max_mparams=20.0),
    convergence_state=ConvergenceState.UNDERTRAINED,
    training_steps=TrainingStepRange(max_steps=10000),
    min_quant_precision=QuantPrecision.INT5_MLP,
)

# Partial RoPE validity: any context
VALID_partial_rope = TechniqueContext()  # all None = no constraints

# int6 validity: requires QAT or small model to avoid roundtrip gap
VALID_int6_qat = TechniqueContext(
    min_quant_precision=QuantPrecision.INT6,
    # QAT implied by absence of failure mode in context
)

# depth_recurrence invalid with int6
INVALID_depth_recurrence_int6 = TechniqueContext(
    min_quant_precision=QuantPrecision.INT6,  # conflicts when int6 or worse
    # This would be represented as a HARD_INCOMPATIBLE edge, not a validity context
)
```

---

### 1.2 Contextual Validity as First-Class Feature

The int5 MLP pattern (works well-trained, fails undertrained) is modeled as multiple `ContextualValidity` entries on the Technique node, NOT as string conditions on edges.

```python
@dataclass
class ContextualValidity:
    """
    Describes whether a technique is valid, invalid, or partially valid
    in a specific context.

    This is the primary mechanism for representing partial/conditional validity.
    A technique may have multiple ContextualValidity entries covering different
    contexts. During query, ALL entries are checked and the most specific match
    is used.
    """
    validity_status: ValidityStatus  # VALID, INVALID, CONDITIONALLY_VALID

    # The context in which this validity status applies
    context: TechniqueContext

    # Evidence for this validity claim
    evidence: list[EvidenceRef]

    # Human-readable explanation of why this is valid/invalid in this context
    mechanism: str

    # Confidence level for this claim
    confidence: ConfidenceLevel  # HIGH, MEDIUM, LOW

    # If CONDITIONALLY_VALID: what additional constraints apply
    conditions: list[str] = field(default_factory=list)

    # If INVALID: severity of the failure
    failure_severity: FailureSeverity | None = None  # CATASTROPHIC, DEGRADATION, MARGINAL

class ValidityStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    CONDITIONALLY_VALID = "conditionally_valid"
    UNKNOWN = "unknown"

class ConfidenceLevel(Enum):
    HIGH = "high"    # multiple independent sources, quantitative
    MEDIUM = "medium"  # single source, quantitative
    LOW = "low"      # anecdotal, qualitative

class FailureSeverity(Enum):
    CATASTROPHIC = "catastrophic"  # e.g., depth_recurrence + int6: 900x error amplification
    DEGRADATION = "degradation"    # e.g., int5_mlp undertrained: +1.1 BPB gap
    MARGINAL = "marginal"          # e.g., xsa + ttt: small negative interaction
```

---

### 1.3 Edge Types — Separated Mechanisms

The critique correctly identified that `ENABLES` conflates budget reallocation with qualitative enablement, and `CONFLICTS` conflates hard incompatibility with soft redundancy. v2 defines five distinct edge types:

```python
class EdgeType(Enum):
    """
    Five distinct edge types, each capturing a different mechanism:

    BUDGET_REALLOCATES: source frees budget, directly funding target
      e.g., int6_qat saves ~4MB → mlp3x costs ~4MB
      Mechanism: arithmetic budget transfer

    CAPACITY_PROVIDES: source increases headroom/capacity without budget transfer
      e.g., mixed_int5_mlp enables wider model by aggressive compression
      Mechanism: scale-up-via-compression

    QUALITATIVE_ENABLES: source qualitatively enables target (no budget relationship)
      e.g., QAT eliminates roundtrip gap, enabling int6 to work
      Mechanism: removes a precondition barrier

    HARD_INCOMPATIBLE: combining source and target is ALWAYS worse than either alone
      Catastrophic interaction — cannot be resolved by ordering or parameters
      e.g., depth_recurrence + int6_qat: 900x error amplification

    SOFT_REDUNDANT: combining source and target provides less benefit than sum of parts
      Can be resolved by dropping one; net result is still positive but sub-additive
      e.g., xsa + ttt_score_first: both target local context
    """
    BUDGET_REALLOCATES = "budget_reallocates"
    CAPACITY_PROVIDES = "capacity_provides"
    QUALITATIVE_ENABLES = "qualitative_enables"
    HARD_INCOMPATIBLE = "hard_incompatible"
    SOFT_REDUNDANT = "soft_redundant"
```

**Rationale for separation:**

| Original (v1) | v2 Split | Mechanism | Query Use |
|---|---|---|---|
| `ENABLES` (int6→mlp3x) | `BUDGET_REALLOCATES` | int6 saves 4MB; mlp3x costs 4MB; net 0 | "What can I afford with freed budget?" |
| `ENABLES` (mixed_int5_mlp→mlp3x) | `CAPACITY_PROVIDES` | Aggressive compression enables larger model | "What compression strategy maximizes capacity?" |
| `ENABLES` (QAT→int6) | `QUALITATIVE_ENABLES` | QAT removes roundtrip gap, enabling int6 viability | "What enables this technique to work?" |
| `CONFLICTS` (depth_recurrence<>int6) | `HARD_INCOMPATIBLE` | 900x error amplification; never combine | "Hard blocks — must not combine" |
| `CONFLICTS` (xsa<>ttt) | `SOFT_REDUNDANT` | Sub-additive local context targeting | "Sub-optimal combinations — warning only" |
| `CASCADE` (wd→int6) | `QUALITATIVE_ENABLES` | wd controls weight distribution, affecting compressibility | "What improves quant behavior?" |
| `ORTHOGONAL` | retained | No interaction | "Can stack freely" |

---

### 1.4 Dependency Edge — Revised

```python
@dataclass
class DependencyEdge:
    source: str                          # technique ID
    target: str                          # technique ID
    edge_type: EdgeType                   # one of 5 separated types
    strength: float                       # 0.0–1.0, confidence in this relationship
    evidence: list[EvidenceRef]           # PR numbers, record IDs

    # Context in which this edge applies — MUST be a typed TechniqueContext, not strings
    valid_context: TechniqueContext | None = None

    # Human-readable mechanism description
    mechanism: str = ""

    # For BUDGET_REALLOCATES edges: quantified budget transfer
    # These fields are REQUIRED for BUDGET_REALLOCATES edges
    budget_delta_mb: float | None = None   # e.g., -4.0 for int6 saving 4MB
    budget_delta_params: int | None = None  # e.g., +4000000 for mlp3x

    # For QUALITATIVE_ENABLES edges: what precondition is removed
    enables_precondition: str | None = None

    # For HARD_INCOMPATIBLE edges: severity and mechanism
    incompatibility_mechanism: str | None = None
    failure_severity: FailureSeverity | None = None

    # For SOFT_REDUNDANT edges: overlap description
    redundancy_mechanism: str | None = None  # e.g., "both target local context"
```

**Concrete edge examples from the report (v2):**

```python
# BUDGET_REALLOCATES: int6 frees budget, directly funding mlp3x
Edge(
    source="int6_qat", target="mlp3x",
    edge_type=EdgeType.BUDGET_REALLOCATES,
    strength=1.0,
    evidence=[PR70, Record6, Record9, Record10],
    valid_context=TechniqueContext(budget_mb=BudgetRange(max_mb=16)),
    mechanism="int6 saves ~4MB over int8; mlp3x costs ~4MB (+4M params); net size ~0",
    budget_delta_mb=-4.0,
    budget_delta_params=+4000000,
)

# CAPACITY_PROVIDES: mixed_int5_mlp enables wider model via aggressive compression
Edge(
    source="mixed_int5_mlp", target="mlp3x",
    edge_type=EdgeType.CAPACITY_PROVIDES,
    strength=0.9,
    evidence=[PR544],
    valid_context=TechniqueContext(
        budget_mb=BudgetRange(exact_mb=16),
        model_size_mparams=ModelSizeRange(min_mparams=30.0),
    ),
    mechanism="int5 MLP is more aggressive than int6; frees additional ~1-2MB; enables wider model",
)

# QUALITATIVE_ENABLES: QAT eliminates int6 roundtrip gap
Edge(
    source="qat", target="int6_qat",
    edge_type=EdgeType.QUALITATIVE_ENABLES,
    strength=1.0,
    evidence=[Record7, PR70],
    mechanism="QAT STE gradient passes through quantization roundtrip; eliminates +0.048 BPB gap",
    enables_precondition="quant_gap_eliminated",
)

# QUALITATIVE_ENABLES: wd_0_04 improves weight distribution for better zstd compression
Edge(
    source="wd_0_04", target="int6_qat",
    edge_type=EdgeType.QUALITATIVE_ENABLES,
    strength=0.8,
    evidence=[PR375, PR238],
    valid_context=TechniqueContext(compression=CompressionAlgorithm.ZSTD),
    mechanism="Higher WD produces weights that compress better under zstd; extends effective int6 savings",
)

# HARD_INCOMPATIBLE: depth_recurrence amplifies int6 quant error 900x
Edge(
    source="depth_recurrence", target="int6_qat",
    edge_type=EdgeType.HARD_INCOMPATIBLE,
    strength=1.0,
    evidence=[PR363],
    mechanism="Weight-sharing across 3 recurrence cycles amplifies int6 gap from +0.001 to +1.14 BPB",
    incompatibility_mechanism="quant_error_amplification_900x",
    failure_severity=FailureSeverity.CATASTROPHIC,
)

# HARD_INCOMPATIBLE: int4 breaks standard techniques in ternary regime
Edge(
    source="int4", target="xsa",
    edge_type=EdgeType.HARD_INCOMPATIBLE,
    strength=1.0,
    evidence=[PR367],
    mechanism="Standard techniques break in bitnet ternary regime",
    incompatibility_mechanism="bitnet_ternary_incompatible",
    failure_severity=FailureSeverity.CATASTROPHIC,
)

# SOFT_REDUNDANT: xsa and ttt both target local context
Edge(
    source="xsa", target="ttt_score_first",
    edge_type=EdgeType.SOFT_REDUNDANT,
    strength=1.0,
    evidence=[PR303],
    mechanism="XSA and TTT both target local context; combined gain is sub-additive (-0.016 BPB vs sum of parts)",
    redundancy_mechanism="local_context_targeting_overlap",
)
```

---

### 1.5 Technique Node — Revised

```python
class TechniqueCategory(Enum):
    QUANTIZATION = "quantization"
    ARCHITECTURE = "architecture"
    OPTIMIZER = "optimizer"
    EVALUATION = "evaluation"
    DATA = "data"
    TESTTIME = "testtime"

class ValidationStatus(Enum):
    MULTI_SEED_VALIDATED = "multi_seed_validated"  # validated across seeds
    SINGLE_SEED = "single_seed"
    ANECDOTAL = "anecdotal"
    FAILED = "failed"               # proven net negative in ALL contexts
    CONTEXTUAL_FAILURE = "contextual_failure"  # fails in some contexts, valid in others
    RETIRED = "retired"             # superseded

class Transferability(Enum):
    HIGH = "high"        # generalizes across setups
    MEDIUM = "medium"    # requires similar hardware/data
    LOW = "low"          # setup-specific
    CONTEXT_SPECIFIC = "context_specific"  # validity depends entirely on context

@dataclass
class CostModel:
    """How this technique affects model size and throughput."""
    size_delta_mb: float | None = None     # signed MB delta; None = unknown
    param_delta: int | None = None         # signed param count delta
    throughput_cost_ms: float | None = None  # per-step overhead; None = unknown
    throughput_cost_range_ms: tuple[float, float] | None = None  # naive vs efficient impl

    def throughput_cost_range(self) -> tuple[float, float]:
        """Return (optimistic, pessimistic) throughput cost."""
        if self.throughput_cost_range_ms:
            return self.throughput_cost_range_ms
        if self.throughput_cost_ms is not None:
            return (self.throughput_cost_ms, self.throughput_cost_ms)
        return (0.0, 0.0)

@dataclass
class QuantInteraction:
    """How this technique interacts with quantization."""
    quant_type: str | None = None           # "ste", "gptq", "per-row", "per-tensor"
    roundtrip_gap: float | None = None      # BPB gap from quantization roundtrip
    gap_eliminated: bool = False
    mechanism: str | None = None

@dataclass
class FailureMode:
    """Known failure condition for this technique."""
    context: TechniqueContext               # context where this fails
    effect: str                            # human-readable effect
    severity: FailureSeverity
    evidence: list[EvidenceRef]
    mechanism: str | None = None

@dataclass
class Precondition:
    """A condition that must be satisfied before this technique can be applied."""
    type: str                              # "quant_precision", "optimizer", "architecture", "training"
    description: str                        # human-readable
    constraint: TechniqueContext | None = None  # typed constraint, if applicable
    is_required: bool = True
    evidence: list[EvidenceRef] = field(default_factory=list)

@dataclass
class Technique:
    id: str
    name: str
    category: TechniqueCategory

    # Parameters (for parameterized techniques)
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)

    # Cost model
    cost_model: CostModel
    quant_interaction: QuantInteraction | None = None

    # Combinatorial structure — no inline lists that duplicate edges
    preconditions: list[Precondition] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)

    # Validation state
    validation_status: ValidationStatus
    validity_contexts: list[ContextualValidity] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)

    # Metadata
    discovered_in: str
    transferability: Transferability
    notes: str = ""

    # -- Derived helpers (computed, not stored) --

    def is_valid_in_context(self, ctx: TechniqueContext) -> ValidityStatus:
        """
        Check all ContextualValidity entries and return the most specific match.
        Priority: INVALID entries > CONDITIONALLY_VALID > VALID.
        """
        best_match: ContextualValidity | None = None
        for vc in self.validity_contexts:
            if vc.context.matches(ctx):
                if best_match is None or vc.validity_status == ValidityStatus.INVALID:
                    best_match = vc
                    if vc.validity_status == ValidityStatus.INVALID:
                        return INVALID  # early exit for hard invalid
        if best_match:
            return best_match.validity_status
        return ValidityStatus.UNKNOWN

    def get_applicable_edges(self, edge_type: EdgeType | None = None) -> list[DependencyEdge]:
        """Query edges from this technique. Returns filtered list if edge_type specified."""
        # Implementation: query graph index by (source=self.id, edge_type=edge_type)
        ...
```

**Removed from Technique node (vs v1):**
- `conflicts_with: list[ConflictEdge]` — now ONLY queried from the graph edge index
- `requires_mutually: list[str]` — redundant with edge queries
- `is_stackable_with: list[str]` — redundant with edge queries (ORTHOGONAL edges)
- `orthogonal_cluster: str | None` — replaced by ORTHOGONAL edges between individual techniques; the cluster is a query result, not a field

---

### 1.6 Constraint Node — Revised

```python
class ConstraintType(Enum):
    BUDGET_SIZE = "budget_size"      # MB
    THROUGHPUT = "throughput"        # ms/step
    ACCURACY = "accuracy"            # BPB
    LAYER_COUNT = "layer_count"      # integer
    MODEL_SIZE = "model_size"        # M params
    TRAINING_STEPS = "training_steps"
    HARDWARE = "hardware"

@dataclass
class Constraint:
    id: str
    type: ConstraintType
    operator: str                    # "<=", ">=", "==", "range", "in"
    value: float | int | list | tuple  # scalar or range [min, max]
    unit: str                        # "MB", "ms/step", "BPB", "layers", "M_params"

    # Context in which this constraint applies
    context: TechniqueContext | None = None

    # For compound constraints: AND/OR composition
    components: list["Constraint"] | None = None  # for compound constraints
    composition: str | None = "AND"  # "AND" or "OR"

    def evaluate(self, ctx: TechniqueContext) -> bool:
        """Returns True if this constraint is satisfied in the given context."""
        if self.components:
            if self.composition == "AND":
                return all(c.evaluate(ctx) for c in self.components)
            else:
                return any(c.evaluate(ctx) for c in self.components)
        # ... evaluate based on type, operator, value, unit
        ...

@dataclass
class ConstraintPropagationPath:
    """A single step in a constraint propagation trace."""
    source_constraint: Constraint
    edge: DependencyEdge
    derived_constraint: Constraint
    mechanism: str

@dataclass
class PropagationResult:
    original_constraints: list[Constraint]
    implied_constraints: list[Constraint]
    enabled_techniques: list[str]
    disabled_techniques: list[str]
    paths: list[ConstraintPropagationPath]
```

---

## 2. Constraint Propagation — Algorithm Specification

The critique correctly identified that v1 had no propagation algorithm. This section specifies it precisely.

### 2.1 Core Traversal Algorithm

```python
def propagate_constraints(
    initial_constraints: list[Constraint],
    query_context: TechniqueContext,
    graph: "TechniqueGraph",
    direction: str = "downstream",
    max_hops: int = 10,
) -> PropagationResult:
    """
    Constraint propagation via backward-chaining graph traversal.

    Algorithm:
    1. Initialize worklist with initial_constraints
    2. For each constraint C in worklist:
         a. Find all edges where edge.valid_context.matches(query_context)
         b. For each matching edge:
              i.  If edge.edge_type is BUDGET_REALLOCATES:
                    - Derive budget delta on target: target_budget = C.value + edge.budget_delta_mb
                    - Create new constraint: Constraint(type=BUDGET_SIZE, value=target_budget, ...)
                    - Add to derived constraints
              ii. If edge.edge_type is QUALITATIVE_ENABLES:
                    - Derive precondition satisfaction on target
                    - Mark target's precondition as satisfied
              iii.If edge.edge_type is HARD_INCOMPATIBLE:
                    - Add target to disabled_techniques
                    - Do NOT traverse further from disabled nodes
              iv. If edge.edge_type is SOFT_REDUNDANT:
                    - Add warning to output (not a blocker)
                    - Continue traversal (soft redundancy doesn't disable)
         c. Add newly derived constraints to worklist (if not visited)
    3. Handle cycles: maintain visited (constraint_id, technique_id) pairs
    4. Return fixed point: no new constraints derived after full pass

    Args:
        initial_constraints: Starting constraints (e.g., [budget <= 16MB])
        query_context: The technique context to evaluate against (layer count, model size, etc.)
        graph: The technique dependency graph
        direction: "downstream" (from cause to effect) or "upstream" (from effect to cause)
        max_hops: Maximum traversal depth to prevent infinite loops

    Returns:
        PropagationResult with all implied constraints and enabled/disabled techniques
    """

    worklist: list[Constraint] = list(initial_constraints)
    visited: set[tuple[str, str]] = set()  # (constraint_id, technique_id)
    derived_constraints: list[Constraint] = []
    enabled_techniques: list[str] = []
    disabled_techniques: list[str] = []
    paths: list[ConstraintPropagationPath] = []
    soft_warnings: list[tuple[str, str]] = []  # (technique_a, technique_b)

    while worklist:
        C = worklist.pop(0)
        C_key = f"{C.type}_{C.id}"

        # Find all edges whose valid_context matches the query_context
        candidate_edges = graph.get_edges_matching_context(C, query_context)

        for edge in candidate_edges:
            edge_key = (C_key, edge.target)
            if edge_key in visited:
                continue
            visited.add(edge_key)

            if edge.edge_type == EdgeType.HARD_INCOMPATIBLE:
                # Hard block: target cannot be used
                if edge.target not in disabled_techniques:
                    disabled_techniques.append(edge.target)
                # Do not traverse further from a hard-incompatible target
                continue

            elif edge.edge_type == EdgeType.BUDGET_REALLOCATES:
                if C.type == ConstraintType.BUDGET_SIZE and edge.budget_delta_mb is not None:
                    # Derive new budget constraint on target
                    if C.operator == "<=":
                        new_value = C.value + edge.budget_delta_mb
                    else:
                        new_value = C.value  # pass through for other operators
                    derived = Constraint(
                        id=f"{C.id}_via_{edge.source}",
                        type=ConstraintType.BUDGET_SIZE,
                        operator="<=",
                        value=new_value,
                        unit=C.unit,
                        context=query_context,
                    )
                    if derived not in derived_constraints:
                        derived_constraints.append(derived)
                        worklist.append(derived)

                    path = ConstraintPropagationPath(
                        source_constraint=C,
                        edge=edge,
                        derived_constraint=derived,
                        mechanism=f"budget_reallocates: {edge.source} delta={edge.budget_delta_mb}MB → {edge.target}",
                    )
                    paths.append(path)

                    if edge.target not in enabled_techniques:
                        enabled_techniques.append(edge.target)

            elif edge.edge_type == EdgeType.QUALITATIVE_ENABLES:
                # Qualitatively enables: mark that a precondition is satisfied
                if edge.enables_precondition and edge.target not in enabled_techniques:
                    enabled_techniques.append(edge.target)
                # Continue traversal: enablement may cascade
                # (e.g., QAT enables int6, int6 enables mlp3x via BUDGET_REALLOCATES)
                continue

            elif edge.edge_type == EdgeType.CAPACITY_PROVIDES:
                # Capacity provision: target is enabled if budget and size constraints permit
                if edge.target not in enabled_techniques:
                    enabled_techniques.append(edge.target)
                continue

            elif edge.edge_type == EdgeType.SOFT_REDUNDANT:
                # Soft warning: record it but don't block
                soft_warnings.append((edge.source, edge.target))

            # Check hop limit
            if len(paths) > max_hops:
                break

    return PropagationResult(
        original_constraints=initial_constraints,
        implied_constraints=derived_constraints,
        enabled_techniques=enabled_techniques,
        disabled_techniques=disabled_techniques,
        paths=paths,
    )
```

### 2.2 Budget Propagation Worked Example

```python
# Query context
ctx = TechniqueContext(
    layers=LayerRange(exact=11),
    model_size_mparams=ModelSizeRange(exact_mparams=33.6),
    budget_mb=BudgetRange(max_mb=16),
    convergence_state=ConvergenceState.WELL_TRAINED,
    training_steps=TrainingStepRange(min_steps=15000),
)

# Initial constraint: budget <= 16MB
initial = [Constraint(id="budget_16mb", type=ConstraintType.BUDGET_SIZE,
                      operator="<=", value=16, unit="MB")]

result = propagate_constraints(initial, ctx, graph)

# Expected derivation:
# Step 1: budget_16mb
#   → BUDGET_REALLOCATES (int6_qat → mlp3x): budget_16mb + (-4MB) = 12MB on mlp3x
#   → BUDGET_REALLOCATES (int6_qat → int6_qat self): not applicable
#   → QUALITATIVE_ENABLES (qat → int6_qat): int6_qat precondition "QAT required" satisfied
# Step 2: derived budget_12mb on mlp3x
#   → mlp3x is enabled
#   → No further BUDGET_REALLOCATES from mlp3x
# Step 3: int6_qat enabled via QUALITATIVE_ENABLES from qat
#   → int6_qat is enabled
#   → BUDGET_REALLOCATES from int6_qat already explored

# Enabled: [int6_qat, mlp3x]
# Disabled: [int4 (not in graph for this context), depth_recurrence (HARD_INCOMPATIBLE)]
```

---

## 3. Query Interface

### 3.1 Query: Addable Techniques

```python
@dataclass
class TechniqueSuggestion:
    technique: Technique
    expected_bpb_delta: float
    throughput_cost_optimistic_ms: float
    throughput_cost_pessimistic_ms: float
    throughput_adjusted_bpb_delta: float  # bpb_delta - (overhead_ms * cost_per_ms)
    reason: str
    confidence: ConfidenceLevel
    preconditions_met: list[Precondition]
    preconditions_missing: list[Precondition]
    validity_in_context: ValidityStatus
    edge_origin: list[DependencyEdge]  # which edges enabled this suggestion

def query_addable_techniques(
    current_stack: list[str],
    query_context: TechniqueContext,
    constraints: list[Constraint] | None = None,
    max_results: int = 10,
    graph: "TechniqueGraph",
) -> list[TechniqueSuggestion]:
    """
    1. Get all techniques not in current_stack
    2. Filter out HARD_INCOMPATIBLE conflicts with any stack member
    3. Filter out techniques INVALID in query_context (via ContextualValidity)
    4. Filter out techniques whose required preconditions are not satisfied
    5. Run constraint propagation to find enabled techniques
    6. Score by throughput-adjusted BPB improvement
    7. Return ranked suggestions

    Throughput adjustment uses the PESSIMISTIC throughput cost (naive implementation)
    unless graph has implementation quality metadata:
      adjusted_bpb = bpb_delta - (throughput_cost_ms * per_ms_cost)

    where per_ms_cost = 0.006 BPB/ms (from PR#375: 86ms/step baseline,
    every 1ms overhead costs ~0.006 BPB).
    """
    candidates = [t for t in graph.get_all_techniques() if t.id not in current_stack]
    suggestions = []

    # Run constraint propagation first
    if constraints:
        prop_result = propagate_constraints(constraints, query_context, graph)
    else:
        prop_result = PropagationResult([], [], [], [], [])

    for technique in candidates:
        # Check contextual validity
        validity = technique.is_valid_in_context(query_context)
        if validity == ValidityStatus.INVALID:
            continue

        # Check hard conflicts with current stack
        hard_conflicts = []
        for stack_tech_id in current_stack:
            edge = graph.get_edge(stack_tech_id, technique.id)
            if edge and edge.edge_type == EdgeType.HARD_INCOMPATIBLE:
                hard_conflicts.append(edge)
        if hard_conflicts:
            continue  # hard block

        # Check preconditions
        preconditions_met = []
        preconditions_missing = []
        for precond in technique.preconditions:
            if precond.constraint and precond.constraint.matches(query_context):
                preconditions_met.append(precond)
            elif precond.is_required:
                preconditions_missing.append(precond)

        # Compute score
        cost = technique.cost_model
        bpb_delta = _compute_bpb_delta(technique, query_context)  # from evidence
        tp_opt, tp_pess = cost.throughput_cost_range()
        per_ms_cost = 0.006  # BPB per ms, from PR#375
        adjusted = bpb_delta - (tp_pess * per_ms_cost)

        suggestions.append(TechniqueSuggestion(
            technique=technique,
            expected_bpb_delta=bpb_delta,
            throughput_cost_optimistic_ms=tp_opt,
            throughput_cost_pessimistic_ms=tp_pess,
            throughput_adjusted_bpb_delta=adjusted,
            reason=_build_reason(technique, prop_result),
            confidence=_compute_confidence(technique),
            preconditions_met=preconditions_met,
            preconditions_missing=preconditions_missing,
            validity_in_context=validity,
            edge_origin=prop_result.paths,
        ))

    suggestions.sort(key=lambda s: s.throughput_adjusted_bpb_delta)
    return suggestions[:max_results]
```

### 3.2 Query: Conflicts

```python
@dataclass
class ConflictReport:
    conflicting_technique: str
    edge_type: EdgeType  # HARD_INCOMPATIBLE or SOFT_REDUNDANT
    strength: float
    evidence: list[EvidenceRef]
    mechanism: str
    failure_severity: FailureSeverity | None = None
    suggestion: str | None = None

def query_conflicts(
    proposed_technique: str,
    current_stack: list[str],
    query_context: TechniqueContext,
    graph: "TechniqueGraph",
) -> list[ConflictReport]:
    """
    Find all conflicts (hard and soft) between proposed technique and current stack.
    Returns them sorted by severity: HARD_INCOMPATIBLE first.
    """
    reports = []
    for stack_tech_id in current_stack:
        edge = graph.get_edge(proposed_technique, stack_tech_id)
        if not edge:
            edge = graph.get_edge(stack_tech_id, proposed_technique)
        if not edge:
            continue

        if edge.edge_type == EdgeType.HARD_INCOMPATIBLE:
            reports.append(ConflictReport(
                conflicting_technique=stack_tech_id,
                edge_type=edge.edge_type,
                strength=edge.strength,
                evidence=edge.evidence,
                mechanism=edge.incompatibility_mechanism or edge.mechanism,
                failure_severity=edge.failure_severity,
                suggestion=_get_incompatibility_suggestion(edge),
            ))
        elif edge.edge_type == EdgeType.SOFT_REDUNDANT:
            reports.append(ConflictReport(
                conflicting_technique=stack_tech_id,
                edge_type=edge.edge_type,
                strength=edge.strength,
                evidence=edge.evidence,
                mechanism=edge.redundancy_mechanism or edge.mechanism,
            ))

    # Sort: HARD_INCOMPATIBLE first, then by strength descending
    reports.sort(key=lambda r: (0 if r.edge_type == EdgeType.HARD_INCOMPATIBLE else 1, -r.strength))
    return reports
```

### 3.3 Query: Orthogonal Clusters

```python
def query_orthogonal_clusters(
    current_stack: list[str],
    query_context: TechniqueContext,
    graph: "TechniqueGraph",
    min_cluster_size: int = 2,
) -> list[OrthogonalCluster]:
    """
    Find groups of techniques that are pairwise ORTHOGONAL (no edges between them)
    and can be freely stacked.

    Algorithm:
    1. Start with techniques not in current_stack that are VALID in query_context
    2. Build a pairwise compatibility matrix using all NON-HARD_INCOMPATIBLE edges
    3. Find maximal cliques or connected components in the compatibility graph
    4. Return clusters with combined BPB and throughput estimates
    """
    candidates = [t for t in graph.get_all_techniques()
                  if t.id not in current_stack
                  and t.is_valid_in_context(query_context) != ValidityStatus.INVALID]

    compatible_pairs: dict[str, set[str]] = {}
    for t in candidates:
        compatible_pairs[t.id] = set()
        for other in candidates:
            if t.id == other.id:
                continue
            edge = graph.get_edge(t.id, other.id)
            if not edge:
                # No edge means unknown (treated as potentially orthogonal)
                compatible_pairs[t.id].add(other.id)
            elif edge.edge_type not in (EdgeType.HARD_INCOMPATIBLE, EdgeType.SOFT_REDUNDANT):
                # ORTHOGONAL or other non-conflicting edge
                compatible_pairs[t.id].add(other.id)

    # Find clusters via connected components on the compatibility graph
    clusters = _find_connected_components(compatible_pairs)
    return [_build_orthogonal_cluster(c, candidates, graph) for c in clusters if len(c) >= min_cluster_size]
```

### 3.4 Query: Ablation Plan Generation

```python
def generate_ablation_plan(
    full_stack: list[str],
    query_context: TechniqueContext,
    graph: "TechniqueGraph",
    control_stack: list[str] | None = None,
) -> AblationPlan:
    """
    Generate ablation plan using technique interaction graph.

    Algorithm:
    1. Build technique interaction matrix from all HARD_INCOMPATIBLE and SOFT_REDUNDANT edges
    2. Identify techniques with ONLY ORTHOGONAL edges to all others:
       → Can be ablated independently (binary search / single removal)
    3. Identify techniques with HARD_INCOMPATIBLE or SOFT_REDUNDANT edges:
       → Require joint ablation with interacting partners
    4. Prioritize by absolute BPB contribution (from evidence)
    5. Limit total runs using greedy set cover until budget exhausted

    For N techniques with all-orthogonal edges: log2(N) ablations suffice (binary search).
    For techniques with interactions: O(2^k) where k = size of interaction cluster.

    Returns plan with estimated runs and ordered steps.
    """
    interaction_matrix = _build_interaction_matrix(full_stack, graph)
    orthogonal_techniques = [t for t in full_stack
                              if t not in interaction_matrix.hard_interactors
                              and t not in interaction_matrix.soft_interactors]
    interacting_clusters = interaction_matrix.get_connected_components()

    steps = []
    # Ablate orthogonal techniques via binary search
    if len(orthogonal_techniques) > 1:
        steps.extend(_binary_search_ablation(orthogonal_techniques, graph))
    # Ablate each interacting cluster exhaustively
    for cluster in interacting_clusters:
        steps.extend(_exhaustive_ablation(cluster, graph))

    return AblationPlan(
        steps=steps,
        estimated_runs=len(steps),
        estimated_total_time=_estimate_time(len(steps)),
        priority_order=_rank_by_bpb_impact(full_stack, graph),
    )
```

---

## 4. QKayV Integration

### 4.1 What Is QKayV

QKayV is the parameter golf research agent being built alongside the TDG. Based on `qkv_parameter_golf_plan.md`, QKayV is a research workflow system that:
- Maintains a stack of applied techniques
- Logs experiment outcomes (BPB, throughput, size)
- Makes decisions about next experiments
- Submits experiment configurations

The TDG v2 is a **library** that QKayV imports and queries. There is no separate service — the TDG is a Python module with a typed API.

### 4.2 Integration Points

**Integration Point 1: QKayV initializes TDG with current context**

```python
# In QKayV's research session initialization
from tdg import TechniqueGraph, TechniqueContext, LayerRange, ModelSizeRange

graph = TechniqueGraph.load("tdg_v1.json")

research_context = TechniqueContext(
    layers=LayerRange(exact=11),
    model_size_mparams=ModelSizeRange(exact_mparams=33.6),
    # ... populated from QKayV's current model config
)
```

**Integration Point 2: Pre-flight conflict checking before experiment submission**

```python
# In QKayV's experiment submission hook
def preflight_check(
    proposed_stack: list[str],
    technique_to_add: str,
    context: TechniqueContext,
) -> PreflightReport:
    """
    Called automatically before an experiment is queued.
    Returns errors (blocking) and warnings ( overridable).
    """
    errors = []
    warnings = []

    # Check hard incompatibilities
    conflict_reports = query_conflicts(technique_to_add, proposed_stack, context, graph)
    for report in conflict_reports:
        if report.edge_type == EdgeType.HARD_INCOMPATIBLE:
            errors.append(PreflightError(
                code="HARD_INCOMPATIBLE",
                message=f"{technique_to_add} + {report.conflicting_technique}: {report.mechanism}",
                severity="ERROR",
                evidence=[str(e.source_id) for e in report.evidence],
            ))
        elif report.edge_type == EdgeType.SOFT_REDUNDANT:
            warnings.append(PreflightWarning(
                code="SOFT_REDUNDANT",
                message=f"{technique_to_add} + {report.conflicting_technique}: {report.mechanism}",
                severity="WARNING",
                can_override=True,
            ))

    # Check contextual validity
    new_tech = graph.get_technique(technique_to_add)
    validity = new_tech.is_valid_in_context(context)
    if validity == ValidityStatus.INVALID:
        errors.append(PreflightError(
            code="CONTEXT_INVALID",
            message=f"{technique_to_add} is INVALID in current context ({context})",
            severity="ERROR",
        ))
    elif validity == ValidityStatus.CONDITIONALLY_VALID:
        warnings.append(PreflightWarning(
            code="CONTEXT_CONDITIONAL",
            message=f"{technique_to_add} is CONDITIONALLY_VALID — check conditions",
            severity="WARNING",
            can_override=True,
        ))

    return PreflightReport(errors=errors, warnings=warnings, info=[])

@dataclass
class PreflightReport:
    errors: list[PreflightError]
    warnings: list[PreflightWarning]
    info: list[str]

@dataclass
class PreflightError:
    code: str
    message: str
    severity: str = "ERROR"
    evidence: list[str] = field(default_factory=list)

@dataclass
class PreflightWarning:
    code: str
    message: str
    severity: str = "WARNING"
    can_override: bool = True
```

**Integration Point 3: Suggesting next experiments**

```python
# In QKayV's suggestion engine
def get_next_experiment_suggestions(
    current_stack: list[str],
    context: TechniqueContext,
    constraints: list[Constraint] | None = None,
    n: int = 5,
) -> list[ExperimentSuggestion]:
    suggestions = query_addable_techniques(
        current_stack=current_stack,
        query_context=context,
        constraints=constraints,
        max_results=n,
        graph=graph,
    )
    return [_to_experiment_suggestion(s) for s in suggestions]
```

**Integration Point 4: Constraint propagation for architecture search**

```python
# In QKayV's architecture search
def explore_architecture_with_constraints(
    target_budget_mb: float,
    target_model_size_mparams: float,
    context: TechniqueContext,
) -> ArchitectureSuggestion:
    constraints = [
        Constraint(id="budget", type=ConstraintType.BUDGET_SIZE,
                   operator="<=", value=target_budget_mb, unit="MB"),
        Constraint(id="model_size", type=ConstraintType.MODEL_SIZE,
                   operator="<=", value=target_model_size_mparams, unit="M_params"),
    ]
    result = propagate_constraints(constraints, context, graph)
    return ArchitectureSuggestion(
        required_techniques=result.enabled_techniques,
        forbidden_techniques=result.disabled_techniques,
        propagation_paths=result.paths,
    )
```

### 4.3 Data Flow

```
QKayV Research Session
    │
    ├── [on init] Load TDG graph + current TechniqueContext
    │
    ├── [pre-flight] preflight_check(stack + new_technique, context)
    │       │
    │       ├── HARD_INCOMPATIBLE? → ERROR (block)
    │       ├── SOFT_REDUNDANT? → WARNING (can override)
    │       └── CONTEXT_INVALID? → ERROR (block)
    │
    ├── [suggestion] query_addable_techniques(stack, context, constraints)
    │       │
    │       ├── Filter by HARD_INCOMPATIBLE
    │       ├── Filter by contextual validity
    │       ├── Propagate constraints
    │       └── Rank by throughput-adjusted BPB
    │
    └── [on experiment complete] Log result to research report
            → TDG updated in next Phase 2 iteration (manual curation)
```

---

## 5. Phase 1 MVP — Concrete Scope and Acceptance Criteria

### 5.1 What Phase 1 Delivers

Phase 1 is scoped to a **single validated technique stack** with full encoding, validation, and querying — not a partial graph of 40 techniques.

**Phase 1 Deliverable: The Foundation Stack**

The stack from Record #14 and Record #11, fully encoded:

```
Foundation Stack:
  - int6_qat (QUANTIZATION)
  - mlp3x (ARCHITECTURE)
  - partial_rope (ARCHITECTURE)
  - ln_scale (ARCHITECTURE)
  - xsa (ARCHITECTURE)
  - wd_0_04 (OPTIMIZER)
  - ema (OPTIMIZER)
  - bigram_hash (ARCHITECTURE)
  - gptq_lite (QUANTIZATION)
```

**Phase 1 Graph Size:**
- ~15 technique nodes (the full set from the report that are MULTI_SEED or SINGLE_SEED validated)
- ~25 edges covering all documented interactions among these 15 techniques
- Full `TechniqueContext` schema on every edge and in every `ContextualValidity` entry
- NO string-based context conditions anywhere in the Phase 1 data

### 5.2 Acceptance Criteria

**AC1: The graph can be loaded and queried without string-matching**
- `TechniqueContext` fields are the only context representation
- Every edge's `valid_context` is a `TechniqueContext` instance (not a list of strings)
- `propagate_constraints` uses only typed fields

**AC2: int5 MLP contextual validity is represented correctly**
- At least two `ContextualValidity` entries on `mixed_int5_mlp`:
  - VALID: well-trained, 15000+ steps, 33.6M params
  - INVALID: undertrained, <10000 steps, <20M params
- `is_valid_in_context()` returns different results for these two contexts

**AC3: Edge types are separated**
- `depth_recurrence <> int6_qat` is `HARD_INCOMPATIBLE` (not CONFLICTS)
- `xsa <> ttt_score_first` is `SOFT_REDUNDANT` (not CONFLICTS)
- `int6_qat <> mlp3x` is `BUDGET_REALLOCATES` (not ENABLES)
- `qat <> int6_qat` is `QUALITATIVE_ENABLES` (not ENABLES)

**AC4: Constraint propagation produces the correct budget cascade**
- Input: `budget <= 16MB`, context = 11 layers, 33.6M params, well-trained
- Output: `int6_qat` and `mlp3x` in `enabled_techniques`
- Output: `depth_recurrence` in `disabled_techniques`
- Output: propagation path explains mechanism (not just a hard-coded example)

**AC5: Pre-flight conflict checking works**
- Submitting `depth_recurrence + int6_qat` together returns an ERROR
- Submitting `xsa + ttt_score_first` together returns a WARNING (not ERROR)
- Submitting `mixed_int5_mlp` at step 8000 (undertrained context) returns an ERROR

**AC6: QKayV integration is functional**
- QKayV can import TDG as a library
- `preflight_check` is callable from QKayV's submission pipeline
- The integration does not require a separate service

**AC7: Query interface returns correct ranked suggestions**
- Given the foundation stack minus bigram_hash, `query_addable_techniques` returns bigram_hash as top suggestion
- Throughput-adjusted BPB uses pessimistic throughput cost by default

### 5.3 What Phase 1 Does NOT Include

- LLM-assisted extraction (Phase 2)
- Real-time learning (Phase 3)
- Graph versioning/rollback (Phase 2)
- Full 40-technique taxonomy (Phase 2 extension)
- Multi-objective Pareto frontier queries (Phase 2)
- Automated contradiction detection (Phase 2)
- Bayesian uncertainty quantification (Phase 3)

---

## 6. Knowledge Acquisition

### 6.1 Manual Extraction Template (Phase 1)

Each PR/record is extracted into typed dataclasses, not strings:

```
PR/Record: PR#303
Techniques: [xsa, ttt_score_first]

Extracted Edge:
  source: xsa
  target: ttt_score_first
  edge_type: SOFT_REDUNDANT
  strength: 1.0
  evidence: [EvidenceRef(source_type="pr", source_id="PR#303", bpb_delta=+0.016, is_positive=False)]
  valid_context: TechniqueContext()  # any context
  redundancy_mechanism: "both target local context"

PR/Record: PR#363
Techniques: [depth_recurrence, int6_qat]

Extracted Edge:
  source: depth_recurrence
  target: int6_qat
  edge_type: HARD_INCOMPATIBLE
  strength: 1.0
  evidence: [EvidenceRef(source_type="pr", source_id="PR#363", bpb_delta=+1.14, is_positive=False)]
  valid_context: TechniqueContext()
  incompatibility_mechanism: "weight-sharing amplifies quant error 900x"
  failure_severity: CATASTROPHIC

PR/Record: PR#544
Technique: mixed_int5_mlp

Extracted TechniqueContextualValidity:
  technique_id: mixed_int5_mlp
  validity_status: VALID
  context: TechniqueContext(
      layers=LayerRange(exact=11),
      model_size_mparams=ModelSizeRange(exact_mparams=33.6),
      convergence_state=ConvergenceState.WELL_TRAINED,
      training_steps=TrainingStepRange(min_steps=15000),
  )
  evidence: [EvidenceRef(source_type="pr", source_id="PR#544", bpb_delta=-0.003, is_positive=True)]
  mechanism: "int5 MLP recovers quality at 33.6M / 15000 steps"
  confidence: HIGH

  [SECOND Entry]
  validity_status: INVALID
  context: TechniqueContext(
      layers=LayerRange(exact=11),
      model_size_mparams=ModelSizeRange(max_mparams=20.0),
      convergence_state=ConvergenceState.UNDERTRAINED,
      training_steps=TrainingStepRange(max_steps=10000),
  )
  evidence: [EvidenceRef(source_type="pr", source_id="PR#238", bpb_delta=+1.1, is_positive=False)]
  mechanism: "undertrained models: +1.1 BPB gap from insufficient compression quality"
  confidence: HIGH
  failure_severity: DEGRADATION
```

### 6.2 LLM Extraction Validation Layer (Phase 2 — designed, not implemented)

```python
@dataclass
class ExtractionCandidate:
    """LLM output before validation."""
    technique_id: str | None
    edge: DependencyEdge | None
    contextual_validity: ContextualValidity | None
    confidence: ConfidenceLevel
    raw_llm_output: str

@dataclass
class ValidationResult:
    candidate: ExtractionCandidate
    status: Literal["auto_merge", "needs_review", "rejected"]
    reason: str
    conflicts_with: list[DependencyEdge] = field(default_factory=list)

def validate_extraction(
    candidate: ExtractionCandidate,
    existing_graph: "TechniqueGraph",
) -> ValidationResult:
    """
    Validation rules:
    1. Check for direct contradictions: same edge, opposite validity status
       → status = "rejected" or "needs_review" depending on confidence comparison
    2. Check for indirect contradictions: new edge conflicts with existing edge
       via shared technique
       → status = "needs_review", list conflicts
    3. Check for consistency: new contextual validity consistent with existing
       → if candidate.confidence == HIGH and no conflicts: "auto_merge"
    4. Check for completeness: required fields populated
       → if missing valid_context on edge: "rejected"
    5. New technique (not in graph): always "needs_review"
    """
    ...
```

---

## 7. Contradiction Handling

### 7.1 Contradiction Data Model

```python
@dataclass
class Contradiction:
    """A detected contradiction between two findings."""
    finding_a: EvidenceRef
    finding_b: EvidenceRef
    techniques: list[str]
    contexts: tuple[TechniqueContext, TechniqueContext]  # the two different contexts
    proposed_resolution: str | None = None
    status: Literal["unresolved", "context_dependent", "genuine_contradiction", "resolved"] = "unresolved"
    resolution_mechanism: str | None = None

    def detect(self) -> bool:
        """
        Returns True if finding_a and finding_b contradict each other
        (opposite validity status for the same technique in similar contexts).

        Algorithm:
        1. If same technique, same context (or context equivalent), different validity: contradiction
        2. If same technique, different contexts, and the contexts are NOT known to produce
           different results: potential genuine contradiction
        3. If same technique, different contexts, and the contexts ARE known to affect validity
           (e.g., training steps): context-dependent, NOT a contradiction
        """
        # Compare contexts — if they differ only in fields known to affect validity
        # (training_steps, model_size, convergence_state), it's context-dependent
        ...
```

### 7.2 int5 MLP Contradiction — How It Enters and Is Resolved

```
PR#544 arrives:
  → Extracted: mixed_int5_mlp VALID at well-trained context
  → Graph updated: ContextualValidity entry added
  → No contradiction

PR#238 arrives (earlier in timeline, but processed after):
  → Extracted: mixed_int5_mlp INVALID at undertrained context
  → Graph updated: second ContextualValidity entry added
  → No contradiction detected: contexts are sufficiently different
    (training_steps differ: 15000+ vs <10000)

System query: "is mixed_int5_mlp valid for 8000 steps?"
  → Checks ContextualValidity entries
  → Finds INVALID entry with training_steps < 10000
  → Returns INVALID with explanation
```

The key fix from v1: the contradiction does NOT need to be "resolved" because both findings are preserved as separate `ContextualValidity` entries with different contexts. The system correctly routes queries to the appropriate entry.

---

## 8. Maintenance

### 8.1 Failure Mode Reporting

```python
def report_failure_mode(
    technique: str,
    failure_context: TechniqueContext,
    mechanism: str,
    severity: FailureSeverity,
    evidence: list[EvidenceRef],
    graph: "TechniqueGraph",
) -> GraphDelta:
    """
    1. Add failure mode to technique.failure_modes
    2. Add or update ContextualValidity with INVALID status for this context
    3. Create HARD_INCOMPATIBLE edges for any techniques that caused the failure
    4. Version the graph delta
    """
    delta = GraphDelta(...)
    graph.apply_delta(delta)
    return delta
```

### 8.2 Graph Versioning

```python
@dataclass
class GraphVersion:
    version_id: str           # semantic version: "1.0.0", "1.1.0"
    timestamp: datetime
    deltas: list[GraphDelta]
    author: str
    message: str

    @staticmethod
    def current() -> "GraphVersion":
        """Load current version from storage."""
        ...

    def rollback(self, target_version: str) -> "GraphVersion":
        """
        Replay deltas in reverse until target_version reached.
        Returns new version representing the rollback.
        """
        ...
```

Storage format: append-only delta log (JSON Lines), with periodic snapshots.
- `/tdg/versions/{version_id}/snapshot.json` — full graph at this version
- `/tdg/versions/{version_id}/deltas.jsonl` — incremental deltas since last snapshot
- Query always reads from latest snapshot + replay deltas

---

## 9. Implementation Roadmap

### Phase 1: Foundation Stack MVP (Weeks 1–2)

**Goal**: Single validated stack, fully typed, pre-flight integration working.

- Implement `TechniqueContext`, `ContextualValidity`, all 5 `EdgeType` variants
- Encode the 15-technique foundation set with ~25 edges
- All edges have typed `valid_context: TechniqueContext | None`
- Implement `propagate_constraints` with the algorithm in Section 2.1
- Implement `query_addable_techniques`, `query_conflicts`, `query_orthogonal_clusters`
- Implement `preflight_check` integration with QKayV
- Acceptance criteria from Section 5.2 must all pass

### Phase 2: Full Taxonomy + Automated Extraction (Weeks 3–6)

- Extend to full 40-technique taxonomy from the report
- Implement LLM extraction pipeline with validation layer
- Implement contradiction detection workflow
- Implement graph versioning with rollback
- Multi-objective query interface (Pareto frontier)
- Evidence confidence scoring

### Phase 3: Real-Time Learning (Weeks 7–12)

- Online learning from QKayV experiment logs
- Bayesian uncertainty on effect estimates
- Cross-domain transfer learning
- User feedback integration
- Causal inference for technique interactions

---

## Appendix: Complete Data Structure Reference

### All Enums

```python
EdgeType = Enum("EdgeType", [
    "BUDGET_REALLOCATES",  # budget transfer: int6 → mlp3x
    "CAPACITY_PROVIDES",   # capacity increase: mixed_int5_mlp → mlp3x
    "QUALITATIVE_ENABLES", # precondition removal: qat → int6
    "HARD_INCOMPATIBLE",   # catastrophic: depth_recurrence + int6
    "SOFT_REDUNDANT",      # sub-additive: xsa + ttt
])
```

### Full Technique Example (int6_qat, v2)

```python
Technique(
    id="int6_qat",
    name="Int6 Quantization via Straight-Through Estimation",
    category=TechniqueCategory.QUANTIZATION,
    parameters={},
    cost_model=CostModel(
        size_delta_mb=-4.0,     # vs int8 on 9-11 layer model
        param_delta=0,
        throughput_cost_ms=0,
    ),
    quant_interaction=QuantInteraction(
        quant_type="ste",
        roundtrip_gap=0.048,
        gap_eliminated=True,
        mechanism="STE gradient passes through quantization roundtrip",
    ),
    preconditions=[
        Precondition(
            type="training_method",
            description="QAT training or small model to avoid roundtrip gap",
            constraint=TechniqueContext(
                convergence_state=ConvergenceState.WELL_TRAINED,
            ),
            is_required=False,
            evidence=[Record7],
        )
    ],
    failure_modes=[],
    validation_status=ValidationStatus.MULTI_SEED_VALIDATED,
    validity_contexts=[
        ContextualValidity(
            validity_status=ValidityStatus.VALID,
            context=TechniqueContext(
                min_quant_precision=QuantPrecision.INT6,
                convergence_state=ConvergenceState.WELL_TRAINED,
            ),
            evidence=[EvidenceRef(source_type="pr", source_id="PR#70", bpb_delta=-0.030, is_positive=True)],
            mechanism="STE QAT eliminates roundtrip gap",
            confidence=ConfidenceLevel.HIGH,
        ),
        ContextualValidity(
            validity_status=ValidityStatus.INVALID,
            context=TechniqueContext(
                min_quant_precision=QuantPrecision.INT6,
                convergence_state=ConvergenceState.UNDERTRAINED,
            ),
            evidence=[EvidenceRef(source_type="record", source_id="Record#6", bpb_delta=+0.048, is_positive=False)],
            mechanism="Without QAT, int6 has +0.048 BPB roundtrip gap",
            confidence=ConfidenceLevel.HIGH,
            failure_severity=FailureSeverity.DEGRADATION,
        ),
    ],
    discovered_in="PR#70",
    transferability=Transferability.HIGH,
    notes="torch.compile dead code trap: late QAT flag can be constant-folded away (PR#315)",
)
```

### Full Edge Example (BUDGET_REALLOCATES: int6 → mlp3x, v2)

```python
DependencyEdge(
    source="int6_qat",
    target="mlp3x",
    edge_type=EdgeType.BUDGET_REALLOCATES,
    strength=1.0,
    evidence=[
        EvidenceRef(source_type="pr", source_id="PR#70", bpb_delta=-0.030, is_positive=True),
        EvidenceRef(source_type="record", source_id="Record#6", is_positive=True),
    ],
    valid_context=TechniqueContext(
        layers=LayerRange(min_inclusive=9, max_inclusive=11),
        budget_mb=BudgetRange(max_mb=16),
    ),
    mechanism="int6 saves ~4MB over int8 on 9-11 layer models; mlp3x adds ~4M params (~4MB at int6); net ~0",
    budget_delta_mb=-4.0,
    budget_delta_params=+4000000,
)
```

### Report-Sourced Interaction Matrix (v2)

| Technique A | Technique B | v2 Edge Type | Evidence | BPB Delta |
|---|---|---|---|---|
| int6_qat | mlp3x | BUDGET_REALLOCATES | PR#70 | -0.030 |
| qat | int6_qat | QUALITATIVE_ENABLES | Record#7 | (eliminates gap) |
| mixed_int5_mlp | mlp3x | CAPACITY_PROVIDES | PR#544 | -0.003 |
| wd_0_04 | int6_qat | QUALITATIVE_ENABLES | PR#375 | (improves compression) |
| depth_recurrence | int6_qat | HARD_INCOMPATIBLE | PR#363 | +1.14 (gap amplification) |
| xsa | ttt_score_first | SOFT_REDUNDANT | PR#303 | +0.016 (negative) |
| int4 | [xsa, smgargate, bigram_hash] | HARD_INCOMPATIBLE | PR#367 | (ternary breaks) |
| int4 | [int6_qat, int8] | HARD_INCOMPATIBLE | PR#367 | +0.065 |
| partial_rope | ln_scale | ORTHOGONAL | Record#14 | -0.0023 |
| xsa | ln_scale | ORTHOGONAL | Record#14 | (no interaction) |
| ema | swa | SOFT_REDUNDANT | PR#287 | ema > swa by 0.003 |
| bigram_hash | mlp3x | ORTHOGONAL | Record#10 | -0.010 combined |
| gptq_lite | int6_qat | ORTHOGONAL | PR#379 | -0.0006 |
| xsa | partial_rope | ORTHOGONAL | Record#14 | (no interaction) |
| mixed_int5_mlp | mlp3x | CAPACITY_PROVIDES | PR#544 | (enables wider model) |
