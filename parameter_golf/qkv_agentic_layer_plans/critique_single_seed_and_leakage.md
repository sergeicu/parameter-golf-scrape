# Critique: Single-Seed Tracker + Leakage Detector Plan

## Overview

The plan demonstrates good awareness of the problems it aims to solve: distinguishing validated findings from single-seed claims, and detecting TTT-related information leakage. The taxonomy of leakage categories is clear, and the statistical framework is sound. However, the plan suffers from significant gaps in implementation detail, internal inconsistencies in statistical thresholds, and an underestimation of complexity in automated log parsing. The "prevention" aspect of leakage detection is largely aspirational rather than implementable. This plan would benefit substantially from a revision pass that addresses the critical gaps identified below.

## Strengths

1. **Clear problem statement**: The plan correctly identifies that single-seed results are being treated as validated knowledge, and that TTT leakage has caused real disqualifications (PR #532).

2. **Well-designed taxonomy**: The 3-category leakage classification (Invalid/Gray Zone/Valid) provides actionable distinctions. The key invariant "every token must be scored under torch.inference_mode() BEFORE any weight update" is a crisp rule.

3. **Statistical framework is mostly sound**: Welch's t-test, bootstrap CIs, and Cohen's d are appropriate choices. The sigma_gap concept (delta/std) for seed recommendation is intuitive.

4. **Good edge case awareness in examples**: The plan correctly notes that PR #528's -0.002 BPB claim is below typical variance and needs multi-seed validation.

5. **Claim vs. evidence tracking**: The ClaimRecord schema properly separates what was claimed from what was validated, with a clear evidence_status enum.

6. **Practical validation thresholds by use case**: The tiered approach (routine ablation=2 seeds, competition=5+ seeds) is appropriate for different consequence levels.

## Weaknesses and Issues

### Part A: Single-Seed Tracker

#### A1. Statistical Threshold Inconsistencies

**Critical inconsistency between lines 173 and 176-179:**

Line 173 states: `Effect size |d| > 0.3 (small effect minimum) OR absolute delta > 0.001 BPB`

But BPB-specific thresholds (lines 176-179) say:
- Meaningful improvement: delta < -0.001 BPB with p < 0.05

These thresholds are in tension. A delta of exactly -0.001 BPB would pass the practical test but likely fails any reasonable effect size calculation given BPB noise levels. More critically, with typical BPB std of 0.002-0.005, a -0.001 BPB delta produces a Cohen's d of only 0.2-0.5 — below the stated |d| > 0.3 threshold. **The plan simultaneously requires |d| > 0.3 AND delta > 0.001 BPB, but these conditions are not jointly satisfiable for small BPB improvements.**

The competition submission threshold (line 178) requires delta < -0.005 BPB AND p < 0.01, which is internally consistent but contradicts the general rule.

**Recommendation**: Clarify which threshold dominates. The effect size and absolute delta criteria should be mutually consistent, or the plan should specify that they operate in different contexts (e.g., effect size for relative comparison, absolute delta for BPB-specific reporting).

#### A2. Mixed-Seed Results Not Handled

The ValidationStatus enum has no state for **mixed results across seeds** (e.g., 3 seeds where 2 show improvement and 1 shows regression). This is a real-world scenario that will occur.

The transitions in Appendix A describe SINGLE_SEED_PENDING moving to MULTI_SEED_PENDING when "all similar" seeds are run, but:
- What if seeds 1 and 3 show -0.002 BPB improvement but seed 2 shows +0.001 BPB regression?
- The current enum would route this to INCONCLUSIVE, which is a catch-all but lacks guidance on the mixed-signals interpretation.
- There's no mechanism to report "2/3 seeds positive, 1/3 negative" as a distinct finding.

**This is under-engineering** — the plan acknowledges variance but doesn't handle bimodal or sign-inconsistent seed distributions.

#### A3. Baseline Requirements Are Underspecified

The statistical tests require a baseline for comparison, but:

- Line 345 checks `if exp.seed_count < 3 or base.seed_count < 3` — but what IS the baseline? In the examples, it varies:
  - PR #528 example uses "Prior record" (1.1233 BPB from PR #379)
  - PR #442 example uses "PR #473 (SGD TTT baseline)" at 1.1214 BPB

**Questions that are unanswered:**
- When comparing two techniques, which baseline is canonical? The plan implies technique comparison but doesn't define baseline hierarchy.
- If I run experiment X on 3 seeds against a SINGLE_SEED_PENDING baseline, does that count as validated?
- Does the baseline need its own multi-seed validation before it can be used to validate other results?

#### A4. The "50+ Reproductions" Validation Path Is Informal

Example 4 (lines 270-289) describes a technique validated through community reproductions rather than multi-seed runs. The plan mentions supporting both paths but:

- There's no schema for recording reproduction events (who, when, conditions)
- "50+ reproductions" is vague — are these independent? Same conditions? Same hardware?
- The ClaimRecord has a `validation_history` field but no mechanism for adding reproduction events
- How does this interact with the ValidationStatus enum? Does reproduction supersede seed count?

#### A5. Reproducibility Counting Is Undefined

The query interfaces reference `count_reproductions(technique_id)` and `reproductions: count_reproductions(technique_id)` but:

- Where do reproduction records come from?
- What counts as a reproduction? Same result on same hardware? Same result different hardware?
- What if 3 teams reproduce a result but with different variances — does that count as 3 reproductions or 1?
- There's no ReproductionRecord schema defined anywhere in the plan.

#### A6. JSONL Storage Will Not Scale

Line 108-115 specifies JSONL storage in `~/.qkv/experiments/`. With thousands of experiments:

- `experiments.jsonl` will become a massive append-only file with no indexing
- No deduplication mechanism (if the same experiment is recorded twice)
- Querying "all techniques with SINGLE_SEED_PENDING" requires scanning the entire file
- No migration path for schema changes (new fields added to ExperimentResult)
- JSONL is not human-editable at scale without tooling

**This is over-engineering in the schema but under-engineering in storage**. The schema is rich and well-thought-out, but the storage format ignores basic scalability concerns.

### Part B: Information Leakage Detector

#### B1. The "Prevention" Claim Is Not Credibly Implementable

Section B5 describes a `LeakagePreventionValidator` that validates experimental setup BEFORE running. The code examples show checking boolean flags like `setup.uses_validation_in_training` and `setup.ttt_validates_before_adapting`.

**The fundamental problem**: These checks require QKayV to have deep knowledge of the training loop's implementation before it runs. But:

- The plan does not specify how QKayV gains access to the training code
- Static analysis of Python code to detect "validation_in_training_loop" is a non-trivial program analysis problem
- Training loops are diverse — there is no standard structure QKayV can rely on
- The example checks at lines 992-998 assume QKayV can inspect the setup object, but never explains how this object is populated

**The "prevention" validator can only work if QKayV:**
1. Has a standardized training loop interface it controls
2. Or performs static code analysis (which is complex and error-prone)
3. Or requires users to annotate their training scripts

None of these are addressed. As written, "prevention" is aspirational marketing copy, not an implementable feature.

**Recommendation**: Rename "Prevention" to "Pre-flight Checks" and make it clear these are checks QKayV CAN perform (e.g., validating a config file against a schema, checking that torch.inference_mode() appears in a code snippet) rather than guarantees against runtime violations.

#### B2. Gray Zone (Category 2) Is Scale-Dependent But Definition Is Static

Line 545 states the problem: "TTT adaptation at token 1001 has already seen tokens from Document A." But lines 550-553 define the gray zone conditions as:
- Processing documents in a merged token stream
- Not using `torch.inference_mode()` during scoring
- Not isolating each document's scoring from subsequent documents

**The issue**: The plan presents Category 2 as a binary state (isolated vs. not), but token-stream leakage is scale-dependent:
- At small scale (few short documents), the information bleed is negligible
- At large scale (millions of tokens), even small per-token leakage accumulates
- The plan provides no threshold for when Category 2 becomes problematic

**What should exist but doesn't:**
- A scale threshold (document count, token count) beyond which isolation is required
- Guidance on how to handle techniques that are Category 2 at small scale but could be Category 3 at large scale
- The relationship between Category 2 severity and the magnitude of the claimed improvement

#### B3. The 3-Category Taxonomy Misses Important Edge Cases

The taxonomy covers:
- Category 1: Direct train on validation
- Category 2: Token-stream TTT without isolation
- Category 3: Document-independent score-first TTT

**Missing cases:**
1. **Leakage through embedding updates during inference**: If embeddings are updated during TTT (even if technically in adaptation phase), this can leak information about validation documents into the model. Category 3's "embeddings frozen" rule addresses this, but the detection logic (Pattern 4, lines 710-733) only marks it as MEDIUM severity.

2. **Cross-seed leakage**: If the same model is adapted on multiple validation documents in sequence without resetting, earlier documents' information could persist. The plan doesn't address multi-document adaptation ordering.

3. **Leakage through normalization statistics**: BatchNorm/LayerNorm running on validation data to compute statistics, even without weight updates. This is a known leakage vector in some architectures.

4. **Gradient checkpointing and memory savings**: Some gradient checkpointing strategies require forward passes on validation data that could theoretically leak.

5. **Learned optimizers**: If the TTT optimizer itself has learned components (e.g., a learned per-parameter learning rate schedule), updating it on validation data constitutes leakage. Not addressed.

#### B4. Detection Patterns Assume Structured Logs

The detection functions (Pattern 1-4) assume experiment logs have structured fields like:
- `step.phase` ("validation", "ttt", "ttt_scoring")
- `step.gradients_computed`
- `step.document_boundaries_respected`
- `step.inference_mode_active`

**Questions unaddressed:**
- What log format generates these structured fields?
- Parameter golf PRs have diverse training scripts — there's no standard logging format
- How does the log parser (Phase 2 deliverable) handle unstructured Python execution?
- What happens when a training script uses its own logging format?

The plan says Phase 2 will build a "log parser for parameter golf training scripts" (line 1183), but:
- This is an entire NLP/code analysis project disguised as a "deliverable"
- No specification of input format
- No specification of how to handle unknown scripts

#### B5. Impact Assessment Is Binary

The `assess_impact` function (lines 1063-1094) returns:
- VALID (no violations)
- INVALID (critical violations)
- DEGRADED (high severity concerns)
- VALID_WITH_CONCERNS (minor violations)

**Problem**: A CATEGORY_1 violation (direct training on validation data) always produces INVALID. But:
- If only 1 of 10 epochs was run on validation data, should that be treated the same as full training on validation data?
- The magnitude of the leakage matters — leaking 10 tokens vs. leaking the entire validation set should have different impacts
- There's no mechanism to quantify the degree of violation, only its presence

### Combined Analysis

#### C1. Shared Infrastructure Assumes Matching Update Patterns

Both features share:
- `~/.qkv/experiments/` storage
- SharedExperimentMetadata schema
- Integration with dependency graph

**Problem**: Part A (single-seed tracking) requires ongoing updates as new seeds are added, new PRs are submitted, and validation status changes. Part B (leakage detection) is largely static per-PR — a PR either has leakage or it doesn't.

- There's no conflict resolution when a new seed contradicts an old leakage assessment
- The dependency graph integration (lines 1143-1146) suggests validation status propagates, but leakage status does too — these could conflict (e.g., technique is VALIDATED_SIGNIFICANT but later discovered to have Category 2 leakage)
- No handling of "retrospective invalidation" — when new information causes QKayV to revise an earlier assessment

#### C2. Phase Timelines Are Unrealistic

**Phase 1 (Weeks 1-2)**: Establish data model, JSONL storage, CLI commands
- This is probably achievable for a skilled developer.

**Phase 2 (Weeks 3-4)**: Build log parser for parameter golf training scripts, automated PR extraction, leakage detector implementation
- **This is not achievable in 2 weeks.** Parsing diverse training scripts to extract structured `phase`, `gradients_computed`, `document_boundaries_respected` fields is a significant program analysis task. Each parameter golf PR has idiosyncratic code structure.

**Phase 3 (Weeks 5-6)**: Seed recommendation engine, follow-up experiment planner, confidence-weighted recommendations
- The seed recommendation engine requires defining optimization objectives (what's the right number of seeds given a budget? Given a desired confidence?)
- Follow-up experiment planning requires understanding hardware configurations, compute costs, and realistic experiment duration estimates

**Total 6 weeks feels optimistic by 2-3x for a team that hasn't built this infrastructure before.**

#### C3. No Handling of Conflict Between Part A and Part B

Consider this scenario:
1. Technique X is recorded with 3 seeds, mean improvement of -0.002 BPB, p=0.04, VALIDATED_SIGNIFICANT
2. Phase 2's leakage detector later flags that the original PR had Category 2 leakage (token stream without document isolation)
3. The technique's status should now be... what?

The plan has no mechanism for:
- Invalidating a VALIDATED_SIGNIFICANT result due to discovered leakage
- Combining validation status with leakage status into a composite confidence
- Reporting "validated but leaky" vs. "validated and clean"

#### C4. The Plan Does Not Address Integration with Existing QKayV

The plan references QKayV's recommendations, dependency graph, and failure mode KB extensively, but:
- How does this actually integrate with the existing codebase? (schema locations, API boundaries, CLI integration points)
- Where does the code actually live? Is it a new QKayV plugin? A new module?
- How do the JSONL files interact with any existing QKayV storage?

This is left entirely to implementation imagination.

## Critical Gaps

The following issues must be addressed before implementation:

1. **Statistical threshold reconciliation**: Lines 173 and 176-179 contain contradictory requirements. Either fix the inconsistency or specify contexts where each applies.

2. **Mixed-seed handling**: The ValidationStatus enum cannot represent "2/3 seeds positive" as a distinct state. This will cause information loss in real usage.

3. **Baseline requirements**: The plan never specifies what makes a valid baseline for comparison, or whether a single-seed baseline can be used to validate multi-seed experiments.

4. **ReproductionRecord schema**: The plan references reproduction counting but has no schema for recording who reproduced, under what conditions, and with what result.

5. **Leakage "prevention" is not implementable**: Section B5 describes a prevention system but provides no mechanism for how QKayV inspects training code. This feature should be renamed and scoped to what is actually achievable.

6. **Log parser specification**: Phase 2 deliverable "log parser for parameter golf training scripts" is underspecified. Without knowing the input format, this deliverable cannot be estimated or implemented.

7. **Scale-dependent Category 2**: Token-stream leakage severity should be tied to document count/token count, not just present/absent.

8. **Storage scalability**: JSONL will not scale to thousands of experiments. Either specify a real database (SQLite, etc.) or accept severe query performance degradation.

9. **Retrospective invalidation**: No mechanism for QKayV to revise an earlier assessment when new information contradicts it.

10. **Composite status**: No specification for how validation status (Part A) and leakage status (Part B) combine into a single technique confidence.

## Recommendations

### Must Fix (Plan Cannot Proceed Without)

1. **Reconcile statistical thresholds**: Ensure effect size (|d| > 0.3) and absolute delta (> 0.001 BPB) criteria are mutually consistent, or specify that they apply in different contexts.

2. **Define baseline requirements**: Specify what constitutes a valid baseline for statistical comparison. Can a SINGLE_SEED_PENDING baseline be used to validate a multi-seed experiment?

3. **Add mixed-seed state to ValidationStatus**: Either add `MIXED_SIGNALS` state or define clear rules for how mixed results across seeds are reported and acted upon.

4. **Define ReproductionRecord schema**: The plan cannot credibly reference reproduction counting without defining what a reproduction record looks like.

5. **Rename "prevention" to "pre-flight"**: The current framing oversells what is implementable. Rename to pre-flight checks and scope them to config validation + basic code pattern matching, not runtime guarantees.

6. **Specify log format for Phase 2**: Either define a standard log format that parameter golf PRs must produce, or acknowledge that automated log parsing is a research problem not solvable in 2 weeks.

### Should Fix (Plan Is Weak Without)

7. **Add retrospective invalidation mechanism**: When new evidence (leakage discovery, new seeds, better baseline) contradicts an earlier assessment, QKayV needs a defined process for revision.

8. **Scale-dependent Category 2 severity**: Specify thresholds (document count, token count) beyond which token-stream leakage becomes unacceptable.

9. **Choose scalable storage**: Either move to SQLite or similar, or explicitly document that this is a prototype system not intended for production scale.

10. **Define composite confidence scoring**: Specify how validation_status and leakage_status combine into a single technique confidence metric.

### Nice To Have (Plan Would Be Stronger With)

11. **Cross-seed adaptation ordering**: Address whether adapting on multiple validation documents in sequence without resetting could cause information persistence.

12. **Gradient checkpointing and normalization leakage**: Document whether these are within scope or explicitly excluded.

13. **Integration with existing QKayV storage**: Specify how this new infrastructure connects to existing QKayV data stores, or confirm this is a greenfield addition.

## Verdict

**Revise**

The plan demonstrates good understanding of the problems and provides a solid conceptual framework. The statistical approach is mostly sound, the leakage taxonomy is clear, and the examples are illustrative. However, the plan has critical gaps that would cause implementation to fail or produce incorrect results:

1. **Internal contradictions in statistical thresholds** will cause inconsistent behavior
2. **Missing mixed-seed state** will lose information in real usage
3. **Underspecified baselines** will lead to incorrect validation conclusions
4. **"Prevention" is not implementable** as described
5. **Log parser specification is absent** making Phase 2 unschedulable
6. **JSONL storage will not scale** to the promised use case

The plan needs a revision pass that addresses the "Must Fix" items before implementation proceeds. The Phase 2 timeline in particular should be re-estimated once the log parser problem is properly scoped. A team starting from this plan would produce a working system that has serious correctness and scalability issues.
