# Critique: Technique Dependency Graph Plan

## Overview

The plan is a structurally ambitious specification for a technique knowledge graph, covering data modeling, knowledge acquisition, querying, integration, and maintenance. It draws correctly from the research report's concrete findings and correctly identifies the core use cases. However, it suffers from **significant specification gaps**: the data model conflates distinct concepts under the same types, the constraint system is under-specified, contradiction handling is promised but not designed, and Phase 1's MVP scope is vague about what "working" means. The plan would benefit from aggressive simplification before any code is written.

---

## Strengths

1. **Ground-truth sourcing is excellent.** The plan correctly anchors every example to specific PRs, records, and issues from the research report. This provenance is essential for a knowledge graph that will be queried for high-stakes experimental decisions.

2. **The 5 core queries (addable techniques, conflicts, orthogonal clusters, constraint propagation, ablation planning) cover the right use cases.** These map directly to the 5 capabilities identified in the research insights: suggesting next experiments, flagging dangerous combinations, and propagating constraints.

3. **Contextual validity is correctly identified as a first-class concern.** The research report contains multiple examples of techniques that work in some contexts but not others (int5 MLP undertrained vs well-trained, depth recurrence with int6 vs int8). Representing this as `ContextualValidity` rather than a simple boolean is correct.

4. **Uncertainty representation is taken seriously.** `UncertaintyBand` and the confidence threshold system (HIGH/MEDIUM/LOW auto-merge) are the right approach.

5. **The 3-phase roadmap is appropriately sequenced.** Manual curation first, then LLM-assisted extraction, then online learning. Phase 1 is the correct foundation.

---

## Weaknesses and Issues

### 1.1 Data Model — `context_conditions` Is Untyped String Soup

The plan defines `context_conditions: list[str]` on edges as `["requires: int6_quant", "only: 11_layers"]`. This is a fundamental specification failure. String-based context conditions cannot be:
- Parsed programmatically without a defined grammar
- Validated for completeness (missing conditions go undetected)
- Extended with new condition types without string-matching code
- Queried efficiently (e.g., "find all edges valid for a 9-layer model")

The research report uses contexts like `"11_layers"`, `"budget: 16MB"`, `"bitnet_ternary"`, `"qat_required"`, `"undertrained"`, `"H100_only"`. These are categorically different: architecture constraints, budget constraints, training-stage constraints, hardware constraints, and convergence-state constraints. A proper `ContextSpec` would define these as typed fields:

```python
@dataclass
class ContextSpec:
    layers: int | tuple[int, int] | None  # e.g., ">=9", or [9, 11]
    model_size_mb: float | None
    quant_precision: list[str] | None      # ["int6", "int8"]
    training_steps: int | None
    convergence_state: str | None          # "undertrained", "well_trained"
    hardware: list[str] | None
    compression: str | None                 # "zstd", "huffman"
```

Without this, the constraint propagation in Section 3.4 cannot be implemented correctly — it will devolve to string matching.

### 1.2 Edge Types Conflate Mechanistically Distinct Relationships

The plan defines 6 edge types, but several are overloaded:

**`CASCADE` covers two fundamentally different mechanisms:**
- `wd_0_04 -> int6_quant` (CASCADE): WD changes weight distribution, which affects compression ratio. This is a qualitative mechanistic relationship.
- `budget_16mb -> int6 -> mlp3x` (CASCADE): Budget forces quantization, which enables MLP expansion. This is a budget allocation path.

These should be separate edge types or structured differently. The first is "influences parameter space"; the second is "enables via resource reallocation."

**`ENABLES` covers at least three distinct mechanisms:**
- `int6 -> mlp3x`: int6 frees ~4MB budget, which funds the MLP expansion. This is budget-mediated.
- `int5_mlp -> mlp3x`: int5 MLP is more aggressive than int6, freeing MORE headroom for wider MLP. Same mechanism, different magnitude.
- `mixed_int5_mlp -> mlp3x` (from PR #544): Train larger model with int5 compression to fit budget while increasing quality. This is a different strategy (scale up model, compress harder).

These have different implications for query answering. "What headroom does int6 provide?" and "What is the most aggressive quantization for a given budget?" are different queries.

**`CONFLICTS` conflates hard incompatibility with soft redundancy:**
- `depth_recurrence <> int6`: Hard incompatibility. Never combine. The 900x error amplification makes the combination strictly worse than either technique alone.
- `xsa <> ttt_score_first`: These are redundantly motivated (both target local context). Combining them is net negative, but not catastrophic. This is a "don't combine" rather than "cannot combine."

A researcher querying "should I add TTT to my XSA stack?" needs different advice than "should I add depth recurrence to my int6 model?" The plan gives both as `CONFLICTS` with `strength=1.0`, which obscures the distinction.

### 1.3 The `orthogonal_cluster` Field Creates Logical Confusion

The plan defines `orthogonal_cluster: str | None` on Technique and `orthogonal_clusters` as a top-level list. The description says techniques in the same cluster are "mutually-compatible" and "combine freely." But this is circular: if techniques are in an `ORTHOGONAL` edge relationship, they're by definition stackable. The orthogonal_cluster appears to be an optimization for "all techniques in this cluster are mutually orthogonal" — but this needs a precise definition of what "orthogonal" means in this context, because the report shows cases where techniques are:
- Mechanistically orthogonal (XSA and LN Scale: different mechanisms)
- Parameter-free orthogonal (Partial RoPE and LN Scale: both zero-cost)
- Conditionally orthogonal (XSA and LN Scale are orthogonal under "any" context, but does this hold for 20-layer models?)

The orthogonal_cluster concept needs a clearer semantic definition or it will create maintenance problems.

### 1.4 The `requires_mutually` and `is_stackable_with` Fields Are Redundant with Edges

The Technique dataclass has both `conflicts_with: list[ConflictEdge]` (edges) and `requires_mutually: list[str]` / `is_stackable_with: list[str]` (inline lists). This duplication creates synchronization risk: if a new CONFLICTS edge is added to the graph, the technique's inline list won't update automatically. The inline lists appear to be denormalization for performance or convenience, but no such motivation is given. Either the inline lists should be removed (always query edges) or the edge store should be derived from them.

### 1.5 Constraint Model Is Incomplete

The `Constraint` dataclass has:
- `type`: BUDGET_SIZE, THROUGHPUT, ACCURACY, HARDWARE, TRAINING_STEPS
- `operator`: "<=", ">=", "==", "range"
- `value`: any
- `unit`: "MB", "ms/step", "BPB", "layers", "epochs"

Problems:
1. **Compound constraints are not representable.** The report frequently describes constraints like "budget <= 16MB AND model >= 20M params" as a single condition for certain techniques. The plan has no compound constraint structure.
2. **The unit system is ad-hoc.** "MB" and "ms/step" and "BPB" have different scales and semantics. Constraint propagation requires understanding that 4MB savings at int6 vs int8 is a quantized precision relationship, not a general arithmetic relationship.
3. **Constraint propagation path in Section 1.3 is not mechanistically derivable.** The example shows `budget_16mb -> ENABLES -> int6_quant -> ENABLES -> mlp3x`, but this path is hard-coded in the example. There is no algorithm defined for how the graph traces this path from a budget constraint. The `derived_constraints` field is a static annotation, not a computed property.

### 2.1 Knowledge Acquisition — Manual Extraction Template Is Too Simple

The extraction template in Section 2.1 captures BPB delta, throughput cost, param count, conflicts, enables, and conditions. But the research report frequently contains:
- **Negative results that are conditional.** PR #303 found XSA + TTT is negative, but only when both are present. The extraction template has no field for "this combination is negative only when both A and B are present, but neutral otherwise."
- **Partial interactions.** PR #375 found MTP is +0.028 BPB due to throughput cost, not because MTP is fundamentally wrong. The extraction template has no field for "this is negative due to throughput cost, not quality."
- **Ordering effects.** Some technique combinations only work in a specific order (e.g., late QAT vs early QAT). The template has no ordering field.

### 2.2 LLM Extraction — No Validation Layer Defined

Section 2.2 promises "a validation layer checks for consistency with existing graph" and "conflicts are flagged for human review." But:
- There is no specification of what consistency means
- No specification of how contradictions between LLM-extracted edges and existing edges are surfaced
- No specification of the human review workflow
- The confidence thresholds (HIGH/MEDIUM/LOW) are defined but not tied to any automatic action beyond queuing

This is a significant gap — Phase 2's core promise is LLM extraction, and the validation layer is the most important and hardest part, yet it is merely referenced, not designed.

### 2.3 Contradiction Handling Is Promised But Not Designed

Section 5.2 defines a `Contradiction` dataclass and resolution rules, but the actual contradiction detection and resolution workflow is not specified. Specifically:
- When does contradiction detection happen? During extraction? During query? During graph merge?
- Who resolves contradictions? A human? An algorithm? What is the algorithm?
- The resolution rules (split into ContextualValidity, flag for human review, deweight lower-confidence finding) are heuristics, not a system

The int5 MLP contradiction (PR #544: -0.003 BPB vs PR #238: +1.1 BPB) is given as an example, but the plan does not explain how this would actually be entered into, detected by, and resolved by the system.

### 3.1 Query Interface — `throughput_adjusted_bpb` Is Physically Incorrect

The plan uses `throughput_adjusted_bpb` as "the real figure of merit" for ranking techniques. The research report (PR #375) found that at 86ms/step, every 1ms of overhead costs ~0.006 BPB. The plan's example computes `throughput_adjusted_bpb = expected_bpb_delta` directly without accounting for the actual per-step time or the actual cost of throughput overhead.

More critically: the plan does not account for the fact that **throughput overhead is not constant for a technique.** XSA with an efficient implementation adds +2ms/step. XSA with a naive implementation adds +7ms/step (from the report: "reducing overhead from 7ms to 2ms per step"). The `throughput_cost_ms` field on Technique is a single float, but the actual cost depends on implementation quality. This will produce wrong rankings.

### 3.2 Query Interface — No Multi-Objective Optimization Support

The research report's central insight is that techniques must be evaluated along 3 axes simultaneously: quality (BPB), size (MB), and throughput (ms/step). The plan's query interface ranks by "expected BPB improvement per unit cost" (Section 3.1) which collapses to a single scalar. But a researcher may want to:
- Minimize BPB regardless of size
- Minimize size regardless of BPB
- Find the Pareto frontier of (BPB, size) tradeoffs
- Find all techniques that improve BPB by >0.005 without increasing size

The plan's `suggest_joint_optimizations` (Section 3.6) partially addresses this but is described only in prose, not as a query API with defined inputs/outputs.

### 3.3 Ablation Plan Generation — Exponential Explosion Not Addressed

The plan acknowledges that "techniques with strong interactions require full grid ablation" but provides no algorithm for determining which technique pairs interact. The report itself contains evidence of interactions (XSA<>TTT, depth_recurrence<>int6), but the plan does not specify how these interaction pairs are represented in the graph or how the ablation planner uses them.

A 10-technique stack has 1024 possible subsets. The plan's ablation plan from Section 3.5 shows only 8 steps, implying a smarter subset selection — but no algorithm is given for how to select these 8 steps from 1024 possibilities.

### 3.4 Constraint Propagation — No Algorithm Specified

Section 3.4 shows the output of `propagate_constraints` but not the algorithm. The example shows a budget constraint cascading through ENABLES edges to find enabled techniques. But:
- What about CASCADE edges? They are described in the data model but not used in the propagation example.
- How does the algorithm handle cycles? (int6 enables mlp3x, and mlp3x... does anything enable int6?)
- How does it handle context conditions during propagation? (If budget=16MB, should the algorithm trace "int6 saves ~4MB on 9-layer models" but not on 20-layer models?)

### 4.1 Integration — QKayV Is Not Defined

The plan references "QKayV's core decision-making" and "QKayV logging" without defining what QKayV is. The research report's plan (`qkv_parameter_golf_plan.md`) describes QKayV as a system being built, but the TDG plan does not define:
- What QKayV's existing architecture looks like
- How the TDG plugs into QKayV (library import? microservice? database?)
- Whether QKayV already has a technique representation that conflicts with TDG's Technique model
- How the pre-flight check integrates with experiment submission

Without this, Phase 1's integration deliverable ("pre-flight conflict checking before experiment submission") cannot be implemented.

---

## Critical Gaps

### Gap 1: The Data Model Cannot Represent Training-State-Dependent Validity

The most nuanced finding in the research report is **int5 MLP's context-dependent validity**: PR #544 found int5 MLP works at 33.6M params / 15000+ steps / well-trained (BPB delta -0.003), while PR #238 found +1.1 BPB gap for undertrained models. The plan's `ContextualValidity` (Section 2.3) has a field `{"training_steps": ">=15000", "model_size": ">=20M"}` but these are string-encoded and not validated.

More critically: **the plan does not specify how to query based on training state.** When a researcher submits an experiment at step 8000, the system should flag that int5 MLP is invalid in their current state. There is no query function for "given my current training state, which techniques are valid?"

### Gap 2: No Specification for How New Techniques Enter the Graph

The plan describes the lifecycle states (NEW, SINGLE_SEED, MULTI_SEED, FAILED, CONTEXTUAL_FAILURE, RETIRED) but does not specify:
- Who assigns the initial lifecycle state when a new technique is discovered
- What transitions are allowed (can something go from FAILED back to SINGLE_SEED? From RETIRED back to NEW?)
- Who can transition states (automated? human review?)
- How RETIRED differs from FAILED in graph query behavior

### Gap 3: No Specification for Negative Evidence

The report contains strong negative results (INT4: +0.065 BPB, depth_recurrence<>int6: +1.14 BPB). The plan's `EvidenceRef` has `is_positive: bool`, implying negative findings can be stored. But:
- A CONFLICTS edge with strength=1.0 and negative BPB delta is different from "this technique was attempted and failed" — the plan does not distinguish these.
- Failed techniques (INT4, MTP) are not in the technique taxonomy table as FAILED; they're listed as NEGATIVE. This is ambiguous in the plan.

### Gap 4: Budget and Model Size Are Not First-Class Citizens

The 16MB budget is the central constraint of the entire competition. The plan treats it as a context condition on edges, but:
- The actual constraint system is "16MB compressed artifact size"
- This depends on model architecture (layer count, hidden dimension), quantization precision, compression algorithm, and weight distribution
- A technique that saves 4MB on a 9-layer model might save different amounts on an 11-layer model
- The plan has no `ModelArchitecture` or `ArtifactSpec` concept

### Gap 5: No Versioning Implementation

Section 5.3 defines `GraphVersion` and `GraphDelta` dataclasses but provides no implementation guidance. The plan promises versioning but does not specify:
- Storage format for versions (append-only log? Git-style DAG?)
- How queries interact with versions (query current? query historical?)
- How rollback actually works (revert to previous JSON? replay deltas?)

---

## Specific Questions Answered

**Is the data model rich enough to capture the nuance in the parameter golf findings?**
Partially. The `ContextualValidity` structure correctly identifies training-state-dependent validity for int5 MLP. However, the string-based `context_conditions` throughout the plan (on edges, in examples) undermine the typed model. The model cannot capture that the same edge (int6 -> mlp3x) has different strength depending on layer count, because `context_conditions` are string tags, not structured constraints.

**Can the graph handle "context-dependent" validity?**
The data model can represent it (ContextualValidity), but the query interface has no function for "given my current context, what techniques are valid?" The only query that touches validity is `query_addable_techniques`, which filters by hard conflicts and preconditions, not by contextual validity ranges.

**How does the graph handle contradictions between PRs?**
It defines a `Contradiction` dataclass but does not specify contradiction detection, resolution algorithms, or the workflow. The int5 MLP contradiction (PR #544 vs PR #238) is described as an example output but not as a process.

**Is the query interface sufficient for the 5 core capabilities?**
The 5 core capabilities are: (1) suggest next experiments, (2) flag dangerous combinations, (3) propagate constraints, (4) find orthogonal clusters, (5) generate ablation plans. All 5 are addressed, but:
- Capability 1 (suggestions) uses a single scalar ranking that doesn't handle multi-objective tradeoffs
- Capability 3 (constraint propagation) has no algorithm specified
- Capability 5 (ablation) doesn't address exponential explosion

**Is the maintenance protocol realistic?**
Partially. The failure mode reporting protocol (Section 5.1) is concrete and actionable. The contradiction handling (Section 5.2) and versioning (Section 5.3) are specified as data structures but lack operational definitions. The claim that "all online updates are treated as lower-confidence until validated" (Phase 3) is good practice but has no implementation.

**Does Phase 1 actually deliver a working MVP?**
Unclear. Phase 1 deliverables are: ~40 technique nodes, ~60 edges, 3 query functions, and pre-flight integration. But:
- There is no acceptance criteria for "working"
- The constraint propagation deliverable mentions only "budget -> int6 -> mlp3x" cascade, implying other constraints are not implemented
- Pre-flight integration with QKayV is undefined (QKayV is not defined in this plan)
- The plan does not define how to validate that the extracted graph matches the report

---

## Recommendations

### R1: Define a Context Schema Before Building Anything Else

The single most important missing specification is a typed, versioned context schema. Define `ContextSpec` as a proper class with typed fields (layer count, model size, training steps, hardware, etc.) before implementing any graph queries. Every `context_conditions` field in the plan should reference this schema. Without this, the constraint propagation system cannot be implemented correctly.

### R2: Separate Budget Constraints from Mechanistic Constraints

`ENABLES` edges that represent budget reallocation (int6 saves 4MB -> mlp3x) should be structured differently from `ENABLES` edges that represent mechanistic enablement. One approach: make budget a first-class `Constraint` node that participates in propagation, rather than encoding budget relationships as edges between techniques.

### R3: Distinguish Hard Conflicts from Soft Redundancy

Change `EdgeType.CONFLICTS` to two types: `INCOMPATIBLE` (hard — combining is always worse than either alone) and `REDUNDANT` (soft — combining provides less benefit than sum of parts). PR #303 (XSA+TTT) and PR #363 (depth_recurrence+int6) belong in different categories.

### R4: Specify the Query Algorithm for Constraint Propagation

Define propagation as a graph traversal with these rules:
1. Start from initial constraints
2. For each constraint C, find all edges where `edge.context_conditions` matches C
3. For each matching edge, derive the implied constraint on the target technique
4. Repeat until fixed point
5. Handle cycles by tracking visited paths

This is a standard backward-chaining algorithm, but it must be explicitly designed.

### R5: Define QKayV Integration Points Before Phase 1

Phase 1's integration deliverable requires knowing what QKayV is and how experiments are submitted. Either define QKayV in this plan or restrict Phase 1 to "offline graph" with no integration, and define integration as Phase 2.

### R6: Implement Ablation Plan Generation as a Separate Pass

The ablation plan should be generated by:
1. Building a technique interaction matrix from all ORTHOGONAL and REDUNDANT edges
2. Techniques with only ORTHOGONAL edges to all others can be ablated independently (binary search)
3. Techniques with REDUNDANT edges require joint ablation
4. Prioritize by absolute BPB contribution (from evidence)

This is implementable. The current prose description is insufficient.

### R7: Add Negative Techniques as First-Class Nodes

Techniques like INT4 and MTP that are proven negative should be in the graph as `validation_status=FAILED` nodes, not absent from the technique list. This enables the "do not try" query that the research plan explicitly calls for.

### R8: Phase 1 Should Deliver a Validated Subgraph, Not a Full Graph

Instead of promising ~40 techniques and ~60 edges, Phase 1 should focus on delivering 3-5 technique stacks with fully validated context-dependent validity. For example, the "foundation stack" from Record #14 (11L + Partial RoPE + LN Scale + EMA + XSA4) could be the Phase 1 deliverable — fully encoded, fully queried, with all context conditions correct. Then Phase 2 extends to the full taxonomy.

---

## Verdict

**Revise.**

The plan correctly identifies the problem space and covers the right capabilities, but it conflates distinct mechanisms under the same types, underspecifies the constraint system, and promises integration with an undefined system. The most critical gap is the string-based `context_conditions` — this single issue invalidates the constraint propagation system and most of the query interface. The plan should not proceed to implementation until a typed context schema is defined and the contradiction detection workflow is designed. Phase 1's scope should be reduced to a single validated technique stack, not a full graph.

The plan is a good specification of the problem space but a poor specification for implementation. It needs significant tightening of the type system and explicit algorithm definitions before code is written.
