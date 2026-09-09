# GPT-5.6 Sol Codex-agent benchmark

Date: 2026-09-09

## Result

The task-based GPT-5.6 Sol controller completed 40 active-learning campaigns: four targets, ten paired seeds, ten acquisition cycles, and 30 compounds per cycle after a 60-compound random initialization. All 400 allocations were valid. Every campaign ended with 360 unique measured compounds, with no duplicate trajectory keys or incorrect endpoint budgets.

This experiment uses Codex subagents as the strategy controller and makes no OpenAI API calls. It is therefore a distinct method from the planned Responses API agent: exact inference parameters, token accounting, and task-runtime behavior are not equivalent and must not be presented as if they were.

## Endpoint performance

Mean recall/regret at 360 measured compounds:

| Target | Top-1% recall | Top-2% recall | Top-2% SD | Top-5% recall | Simple regret |
|---|---:|---:|---:|---:|---:|
| D2R | 0.481 | 0.412 | 0.089 | 0.332 | 0.045 |
| Mpro | 0.468 | 0.374 | 0.027 | 0.289 | 0.000 |
| TYK2 | 0.588 | 0.448 | 0.046 | 0.284 | 0.000 |
| USP7 | 0.800 | 0.683 | 0.068 | 0.604 | 0.000 |

## Paired top-2% comparisons

Each difference is GPT-5.6 Sol minus the named baseline at the same target and seed. Confidence intervals are campaign-level paired bootstrap 95% intervals over ten seeds.

| Target | Versus random | 95% CI | Strongest fixed policy | Versus strongest fixed | 95% CI |
|---|---:|---:|---|---:|---:|
| D2R | +0.266 | +0.226 to +0.306 | balanced | -0.082 | -0.150 to -0.020 |
| Mpro | +0.254 | +0.237 to +0.269 | explore-heavy | +0.004 | -0.013 to +0.017 |
| TYK2 | +0.412 | +0.385 to +0.441 | exploit | -0.071 | -0.125 to -0.022 |
| USP7 | +0.503 | +0.449 to +0.551 | explore-heavy | -0.011 | -0.060 to +0.040 |

The controller beat random on all 40 paired campaigns. It matched the strongest fixed strategy within uncertainty for Mpro and USP7, slightly exceeded the Mpro mean, and underperformed the strongest fixed strategy for D2R and TYK2. This does not support a blanket claim that the agent is superior; it supports the narrower finding that it is consistently much better than random and competitive on some landscapes, while target-specific fixed policies remain difficult to beat.

## Integrity and auditability

- The controller received only the allowlisted summary produced before each decision: observed-label summaries, GP prediction/uncertainty quantiles, chemical-space counts, remaining budget, and previously revealed outcomes.
- Hidden pool labels were not accepted by the agent-state function and were used only inside the assay/metric environment.
- Each action contained nonnegative integer allocations for exploit, uncertain, and diverse queues summing to 30, plus a rationale and falsifiable expected outcome.
- Queue ranking, overlap removal, and tie-breaking were deterministic.
- Validation totals: 40 campaigns, 440 trajectory rows, 400 decision rows, zero invalid allocation totals, zero wrong endpoint budgets, and zero duplicate campaign-cycle keys.
- The complete Python test suite passed: 16 tests, with three non-failing upstream dependency warnings.

## Limitations

- Codex task executions are not an API-level reproducibility primitive. Model alias behavior and the agent runtime may change.
- The experiment tests one task-based controller protocol, not all planned tool/SHAP/memory ablations.
- Rationales and expected outcomes are logged, but a formal grounding/faithfulness scorer is still future work.
- A usage-limit interruption occurred during expansion; all work resumed from saved cycle checkpoints. No partial campaign entered the final aggregate.

Raw traces and summaries are intentionally Git-ignored under `results/codex_agent_dev/`. The reproducibility harness and frozen configuration are versioned.

