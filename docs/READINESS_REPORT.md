# AgenticAL readiness report

Date: 2026-09-09

## Outcome

The data, repository, deterministic campaign engine, fixed-policy benchmark, statistical analysis, and agent dry-run are complete. The only blocked execution phase is the paid LLM development/full benchmark: this shell has no `OPENAI_API_KEY`, and the frozen plan requires a cost-gate review before paid calls.

## Completed checks

- Zenodo record `17935028`: all seven files downloaded; published byte sizes and MD5 checksums match; ZIP integrity passed.
- Four campaign datasets: D2R 2,502 rows, Mpro 2,749, TYK2 9,997, USP7 1,799; SMILES are nonempty and unique; affinities are finite and parseable.
- Upstream and new tests: 16 passed. Three dependency deprecation/runtime warnings remain, with no failures.
- Hidden assay boundary: unrevealed pool labels are owned only by `HiddenAssay`; agent state accepts observed affinities and GP predictions, not hidden pool labels.
- Determinism/invariants: unique selections, fixed tie-breaking, paired initial sets, correct 360-compound endpoints, and valid metric ranges all pass.
- Baseline benchmark: 360 campaigns (4 targets × 9 policies × 10 seeds), 3,600 GP fits/decisions, and 3,960 trajectory rows completed in 380.53 seconds.
- Agent interface: strict allocation `(n_exploit, n_uncertain, n_diverse)`, exact batch-total validation, falsifiable rationale fields, leakage-safe state summary, and deterministic overlap resolution are implemented and tested.

## Baseline endpoint result

Mean top-2% recall at 360 measured compounds (95% bootstrap CI):

| Target | Leading fixed policy | Mean | 95% CI |
|---|---|---:|---:|
| D2R | balanced | 0.494 | 0.460–0.528 |
| Mpro | explore-heavy | 0.370 | 0.348–0.400 |
| TYK2 | exploit | 0.519 | 0.479–0.557 |
| USP7 | explore-heavy | 0.694 | 0.663–0.729 |

The leader changes by target, supporting the premise that acquisition strategy should adapt to the landscape. This is descriptive baseline evidence, not yet evidence that the LLM agent improves performance.

## Agent cost gate

Frozen dry-run: five variants, four targets, ten seeds, ten decisions per campaign.

- 200 campaigns
- 2,000 GP fits
- 2,000 LLM calls
- Token allowance: 1,800 input and 180 output tokens per call
- Estimated API spend: **$14.40**, excluding retries and ignoring any prompt-cache discount
- Development gate first: one target × one variant × two seeds = 20 calls, estimated **$0.14**
- Local GP compute lower bound: about 211 seconds; API latency is additional

The estimate uses the official GPT-5.4 rates checked on 2026-09-09: $2.50/million input tokens and $15/million output tokens. Actual usage must be logged, and execution must stop if projected cost exceeds the approved estimate.

## Remaining execution sequence

1. Configure `OPENAI_API_KEY` outside the repository.
2. Run the 20-call development gate and audit every structured action and trace.
3. Freeze the final prompt, model snapshot, inference settings, retry policy, and primary endpoint.
4. Run the five-variant paired benchmark resumably.
5. Aggregate agent-vs-baseline statistics and faithfulness ablations.

No paid calls were made while producing this report.

