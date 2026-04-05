# Critique: Failure Mode KB Plan

## Overview

The plan is a reasonable starting point for a Failure Mode Knowledge Base but suffers from significant gaps in condition modeling, circular dependency risks with the graph, incomplete handling of failure overturning, and underspecification of the query engine. The specific failure entries in Section 6 are well-formed, but the generalizable infrastructure to support them at scale is weak. The plan reads as an aspirational specification rather than an implementation-ready design.

**Verdict: Revise** — The core concept is sound, but fundamental data model and architecture decisions require redesign before implementation.

---

## Strengths

1. **Mechanism-focused design**: The emphasis on `root_cause_mechanism` over surface-level "what failed" is correct. This enables generalization, which is the actual value of a KB over a simple bug list.

2. **Severity and confidence scoring**: The two-dimensional risk assessment (severity impact × confidence) provides useful triage semantics.

3. **Partial failure handling (Section 3.3)**: The conditional entry pattern for `int5 MLP quantization` is the right approach for multi-regime failures. However, this pattern is not integrated into the main schema.

4. **Specific failure entries (Section 6)**: The 10 documented failures are concrete, well-annotated, and demonstrate genuine understanding of the failure modes. FM-006 (Depth Recurrence + int6) is particularly insightful.

5. **Query interface breadth**: Five distinct query types cover the major use cases, from single-technique lookup to full-stack safety checks.

6. **Knowledge acquisition priority tiers**: The P0-P3 source prioritization is sensible and shows understanding of extraction cost/quality tradeoffs.

---

## Weaknesses and Issues

### 2.1 Primary Schema — Critical Gaps

**Problem: `triggering_conditions` is underspecified**

The `conditions` block uses natural language strings where machine-queryable structured data is needed:

```yaml
conditions:
  scale_range: string | null  # e.g., "< 100M params"
  training_duration: string | null  # e.g., "< 10K steps"
```

`< 100M params` is a string, not a queryable predicate. When a user asks "will this technique fail at 50M params?", the query engine must parse arbitrary strings like `< 100M params`, `> 1B params`, `100M-500M params`. This is an AI-complete problem. The schema should use structured bounds:

```yaml
conditions:
  scale_range:
    min_params: null  # exclusive
    max_params: 100_000_000  # exclusive
    min_inclusive: false
    max_inclusive: false
  training_duration:
    min_steps: null
    max_steps: 10_000
```

**Problem: `constraint_type` enum is insufficient**

```yaml
constraint_type: enum[time_constrained, size_constrained, memory_constrained, compute_unconstrained]
```

Real experiments have **multiple simultaneous constraints**. A 10-minute budget on H100 with 8GB memory is *both* time-constrained and memory-constrained. The enum cannot represent this. Should be:

```yaml
constraints:
  time_constrained: true
  time_budget_minutes: 10
  memory_constrained: true
  memory_budget_mb: 8192
  compute_unconstrained: false
```

**Problem: `outcome_unit` is not generic**

The schema assumes BPB as the universal metric. What about compression ratio, perplexity, throughput (tokens/sec), memory usage, or training time? The `outcome` block should support multiple metric types:

```yaml
outcome:
  primary_metric:
    value: float
    unit: string
    baseline_value: float
    delta: float
    direction: enum[negative, positive, neutral]
  secondary_metrics:
    - name: "throughput"
      value: 850
      unit: "tokens/sec"
      delta: -0.12
      direction: "negative"
```

**Problem: `related_failures` by ID is a denormalization risk**

If FM-001 and FM-002 are related, this is stored as ID references. But when FM-001 is deleted or significantly modified, FM-002's `related_failures` becomes stale. There is no referential integrity mechanism described.

**Problem: Missing `experiment_context` block**

Failures are not transferable across all contexts. The following are completely absent:

- **Hyperparameters**: Learning rate, batch size, warmup steps — a technique may fail only at specific LR values
- **Model architecture family**: "Works on decoder-only, fails on encoder-decoder"
- **Loss function**: "Fails when using cross-entropy, works with RL loss"
- **Dataset size**: "Fails on < 10B tokens, works above"
- **Random seed sensitivity**: "High variance across seeds — results not reproducible"

### 2.2 Uncertainty Representation — Incomplete

```yaml
uncertainty:
  mechanism_confidence: enum[known, suspected, unknown]
  condition_boundary: enum[well_bounded, approximate, poorly_constrained]
  outcome_variance: float
  conflicting_evidence: boolean
  conflicting_sources: string[] | null
```

**Problem: No way to represent contested failures**

A failure mode may have strong evidence from Source A and strong counter-evidence from Source B. The current schema can mark `conflicting_evidence: true` but provides no mechanism for resolving the conflict or presenting both views to the user.

**Problem: No temporal uncertainty**

`condition_boundary: approximate` does not capture that the boundary may shift with new hardware (e.g., a technique that failed on H100 may succeed on H200 due to different memory bandwidth).

### 2.3 Integration with Dependency Graph — Circular Dependency Risk

The plan describes bidirectional influence:
- FMKB entries **infer** edges to add to the dependency graph
- Dependency graph is **used during planning** to avoid failures

**This creates a circular dependency**: The graph is used to predict failures, but the FMKB modifies the graph. If the FMKB infers an incorrect edge, it changes the graph, which changes failure predictions, which generates new FMKB entries, which infer more edges. No mechanism for damping this feedback loop is described.

**Specific problem**: `suggested_edges_to_add` is not distinguished from manually curated graph edges. If an FMKB entry infers an edge that turns out to be wrong (e.g., a false positive conflict), how is it retracted? The plan does not address edge removal.

### 3.1 Source Types — Missing Extraction Quality Metrics

The plan states "LLM extraction with > 80% precision" as a success criterion for Phase 2, but:

- No definition of what "precision" means in this context
- No ground truth dataset to measure against
- No description of how the LLM extraction pipeline works (which model? what prompt engineering?)

### 3.3 Partial Failures — Schema Integration Missing

The partial failure example in Section 3.3 uses a `partial_failure_entry` structure with inline condition contexts, but this pattern is **not present in the main schema** (Section 2.1). The main schema's `conditions` block is a single record, not a list of condition-outcome pairs.

This means:
- Partial failures cannot be stored in the standard format
- The query engine (Section 4) has no defined behavior for partial failures
- The planning algorithm (Section 5.2) does not account for regime-dependent outcomes

### 4.1 Query Functions — Underspecified

**Query A** is the primary use case ("Will technique X fail given my setup?") but the matching logic is not specified:

> "Reasoning: Rules engine + LLM for condition matching against query context"

How does the rules engine evaluate `scale_range: "< 100M params"` against `model_size_params: 50_000_000`? Parsing natural language conditions is not a rules engine problem — it's an NLU problem. If you're using an LLM anyway for condition matching, the "rules engine" component is decorative.

**Query C** ("What throughput overhead threshold must a technique stay below?") is actually computing a derived value:

```
max_allowed_overhead = baseline_throughput * (time_budget / reference_training_steps) - baseline_throughput
```

This is a formula, not a KB lookup. It belongs in the planner, not the KB. Including it as a "query" conflates KB queries with derived calculations.

**Query D** ("Which techniques are known to have negative interactions?") requires the `relationships.conflicts_with` field, but there is no efficient indexing strategy. With 500+ entries, scanning all pairwise combinations is O(n²). No spatial index or graph index is proposed.

### 4.2 Query Engine — Wrong Tool for the Job

SQLite with FTS5 is insufficient for multi-dimensional condition matching:

1. **Multi-dimensional range queries**: "scale > 50M AND scale < 200M AND steps > 10K AND memory < 8GB" requires a proper columnar store with bitmap indexes, not row-based SQLite
2. **Condition matching**: The plan acknowledges that "LLM for complex condition matching" is needed, which means SQLite is just a document store with FTS, not a query engine
3. **Scalability**: At 500+ entries with complex conditions, SQLite will require full table scans for most interesting queries

The plan should commit to either:
- A proper relational schema with indexed columns for each condition dimension, OR
- A vector database with embeddings for semantic condition matching

The hybrid "SQLite + LLM" approach is a sign that neither approach is fully thought through.

### 5.1 Graph Updates — Undefined Edge Confidence and Lifecycle

```yaml
graph_inference:
  suggested_edges_to_add: list[{from, to, type, confidence}]
```

What does `confidence` mean here? Is it the same as the FMKB entry's confidence? Can an FMKB entry with confidence 0.5 infer a graph edge? What if the graph already has this edge with higher confidence from manual curation?

**No answer is given for edge lifecycle**: Who owns graph edges? If FMKB infers an edge and a human later contradicts it, what happens? The plan does not distinguish between:
- FMKB-inferred edges (automated, potentially high error rate)
- Manually curated edges (trusted)
- Hybrid edges (manually approved FMKB inferences)

### 5.2 Planning Algorithm — Severity Threshold is Arbitrary

```python
If failure_modes with severity >= 3: remove candidate, log warning
If failure_modes with severity < 3: include with warning
```

**Why 3?** The severity scale ranges 1-5. A threshold of 3 means "Major" failures block adoption but "Moderate" ones don't. This is a policy decision, not a technical one. The plan should explain how this threshold is calibrated and whether users can override it.

### 6.0 Specific Failures — Inconsistencies and Errors

**FM-001 MTP**: `scale_range: "< 100M params"` but the mechanism describes a time-constrained training issue. The failure is NOT about model scale — it's about training time budget. The `scale_range` field is misleading.

**FM-003 int5 MLP**: States `training_duration: "10,000-20,000 steps"` but notes say "Competitively trained models at 20K steps show no penalty." This is internally contradictory — is 20K in or out?

**FM-007 XSA + TTT**: The `outcome.direction: "negative"` for a combined technique that cancels out two positive individual techniques is correct. But the `relationships` block is empty (`conflicts_with: []`), despite the note saying they "conflict" mechanistically. This is a data inconsistency.

**FM-009 Label Smoothing**: Classified as a "failure" in the plan but the notes say "neutral result, not a failure" and `severity: 1`. The plan is internally inconsistent about whether neutral results belong in a "Failure Mode" KB.

### 7.0 Implementation Roadmap — Missing Milestones

**Phase 1**: "Build basic query API" but no definition of what "basic" means. Which of the 5 queries (A-E) are in scope?

**Phase 2**: "User feedback workflow (report → validate → confirm)" — no definition of the validation process. Who validates? How many confirmations are needed?

**Phase 3**: "Build context-aware failure prediction" — this sounds like ML prediction, but no model is specified. Is it a classifier? A rule-based system? LLM-based?

**Phase 4**: "Community contribution portal" — no specification of contribution guidelines, review process, or conflict resolution.

---

## Critical Gaps

The following issues **must be addressed** before implementation:

### 1. Failure Overturning Protocol

When a technique previously marked as failing is later found to work (e.g., a bug in the original implementation was fixed), the plan provides no protocol for:
- Marking an entry as "overturned" vs. "deleted"
- Preserving the historical record (for auditability)
- Propagating the update to the dependency graph (retracting inferred edges)
- Notification to users who may have acted on the old failure

**Proposed minimum addition**:
```yaml
entry_status: enum[active, overturned, deleted]
overturn_evidence:
  source: string
  explanation: string
  date: timestamp
```

### 2. Multi-Dimensional Condition Indexing

The plan explicitly acknowledges "multi-dimensional condition spaces" in the prompt questions but provides only flat string fields. For real-time decision making, the query engine needs:
- Indexed numeric fields for: model size, training steps, time budget, memory budget
- Indexed categorical fields for: hardware, dataset type, constraint type
- A proper condition matching algorithm (not LLM parsing of strings)

### 3. Circular Dependency Mitigation

The FMKB → Graph → Planning → FMKB feedback loop must be formally analyzed. At minimum:
- Distinguish FMKB-inferred edges (low confidence, needs approval) from curated edges
- Implement edge provenance tracking
- Define conditions under which FMKB-inferred edges are auto-promoted vs. require manual review

### 4. Partial Failure as First-Class Citizen

The partial failure pattern (Section 3.3) must be integrated into the primary schema. The main `conditions` block should support a list of condition-outcome pairs:

```yaml
regime_specific_outcomes:
  - conditions: {training_steps: {<: 10000}, model_scale: {<: 100M}}
    outcome: +0.007 BPB
  - conditions: {training_steps: {>: 20000}, model_scale: {<: 100M}}
    outcome: +0.001 BPB
```

### 5. Experiment Context Fields

Missing from the schema:
- `hyperparameters`: {learning_rate, batch_size, warmup}
- `model_architecture`: {type, num_layers, hidden_size, attention_type}
- `dataset`: {size_tokens, avg_document_length, domain}
- `random_seed`: for reproducibility assessment

### 6. Query Latency Requirements

Phase 1 success criteria states "Query latency < 50ms for single-technique lookup" but no implementation is specified that can achieve this with complex multi-dimensional condition matching. SQLite FTS5 on 10 entries can do this. SQLite on 500 entries with complex joins cannot.

---

## Recommendations

### Immediate Fixes (Before Implementation)

1. **Redesign the condition schema** to use structured numeric bounds instead of natural language strings. This is the single highest-impact change.

2. **Define failure overturning protocol** before writing any code. This is a critical data lifecycle concern.

3. **Separate KB-inferred graph edges from curated edges** with distinct confidence tracks and approval workflows.

4. **Integrate partial failure handling into the main schema** rather than treating it as an aside.

5. **Remove Query C** (throughput budget calculator) from the KB — it is a calculation, not a query, and belongs in the planner.

### Design Clarifications

1. **Commit to a storage technology**: Either use a proper relational DB (PostgreSQL with jsonb, with indexed columns for each condition dimension) or use a vector DB with embeddings for semantic matching. The SQLite + FTS5 + LLM hybrid is not a coherent architecture.

2. **Define "precision" for extraction**: Create a ground truth test set of 20 manually annotated failure reports. Use this to measure LLM extraction precision before building Phase 2.

3. **Specify the planning severity threshold**: Explain the policy rationale for why severity >= 3 blocks adoption. Allow this to be configurable.

4. **Clarify neutral results**: If FM-009 (Label Smoothing, delta=0) is in the KB, it should not be called a "failure." Rename the KB to "Outcome Knowledge Base" or create a separate "neutral_results" section.

### Architecture Improvements

1. **Event sourcing for graph updates**: Rather than direct mutation, use an event log: "FMKB entry added → infer edges → queue for review → edges approved/auto-approved based on confidence." This provides auditability and prevents circular feedback.

2. **Condition matching as a separate service**: Build a dedicated "Condition Evaluator" that takes a technique + context and returns a match score. This can be unit tested independently of the KB storage layer.

3. **Add entry provenance to all fields**: Each field in an entry should track: source_document, extraction_method (manual/LLM), extraction_confidence. This enables post-hoc quality assessment and debugging.

---

## Verdict

**Revise**

The plan is not implementation-ready in its current form. The most critical issues are:

1. **Underspecified condition representation** — the plan acknowledges multi-dimensional conditions but provides only string fields
2. **Circular dependency risk** — FMKB ↔ Graph feedback loop has no damping mechanism
3. **Missing failure lifecycle management** — no protocol for overturning, updating, or deleting entries
4. **Query engine is underspecified** — SQLite + FTS5 + LLM is not an architecture, it's a workaround for not making hard design decisions

The good news: the core concept is sound, the specific failure examples are excellent, and the query interface design (Sections 4.1 A-E) covers real use cases. The plan needs a thorough redesign of the condition data model and the FMKB-Graph integration before coding begins.

A good next step: produce a **technical specification document** that makes hard choices about storage technology, condition matching algorithms, and the FMKB-Graph event protocol — separate from this plan document.
