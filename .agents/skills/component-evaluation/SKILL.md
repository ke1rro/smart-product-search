---
name: component-evaluation
description: Use when defining or executing Smart Product Search component checks, NLP quality measurements, baseline comparisons, or error analysis.
---

# Component evaluation

Keep engineering checks separate from NLP quality evidence. Read
[AGENTS.md](../../../AGENTS.md), the component contract in
[PROJECT_IDEA.md](../../../docs/PROJECT_IDEA.md), and the authorized iteration in
[IMPLEMENTATION_PLAN.md](../../../docs/IMPLEMENTATION_PLAN.md).

## Required inputs

- Component purpose, input/output contract, and current implementation status.
- Authorized evaluation scope and available local fixtures or labeled data.
- Baseline, metric definition, split, model/prompt revision, and relevant settings.

## Procedure

1. Define observable success and failure cases before substantial implementation.
   If a component is not implemented, record it as pending. Do not implement a
   later iteration or make an API call just to fill an evaluation table.
2. Choose evidence for the contract: deterministic configuration/validation tests;
   judged-relevant Recall@100 for retrieval; nDCG@10 on a documented judged candidate
   pool for reranking; query-grouped macro-F1 and class supports for classification;
   ordering/state tests for grouping; real search-to-UI checks for the whole system.
3. Inspect fixtures and prevent leakage. Gold labels remain evaluation-only.
   Keep published train/test boundaries and use seed 2026 only when sampling is
   required. Record selected query IDs. Unjudged products are unknown, not irrelevant.
4. For NLP quality, compare the chosen component with a simple relevant baseline
   on the same data and protocol. State any missing baseline instead of claiming an
   improvement. Engineering test pass counts are not NLP quality scores.
5. Run only authorized checks with `uv run` from the repository root. Keep raw
   results, failed calls, sample counts, settings, and commands. Paid calls, training,
   and speculative model comparisons are outside this workflow's authorization.
6. Review concrete errors and report unsupported cases. When LLM judging is used,
   preserve its rubric, human calibration examples, raw judgments, and limitations.

## Outputs and verification

Write run records under `.artifacts/evaluations/<component>/`, including the exact
command, UTC date, code/model/prompt/data versions, sample size, baseline/candidate
results, failures, and error examples. Mark unexecuted measurements as not run.

For Iteration 0, report configuration test results and engineering checks only;
retrieval, reranking, classification, and end-to-end NLP quality remain unevaluated.
Never report an overall system pass from configuration tests alone.
