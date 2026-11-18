# AgenticAL next-phase experiment plan

Date frozen: 10 September 2026

## Purpose

This phase tests whether the controller's performance and decisions depend on four information channels, measures whether its written forecasts are testable and calibrated, and evaluates a deterministic availability constraint. A versioned OpenAI API comparison is a separate reproducibility check and must not be pooled with Codex-task runs.

## Frozen common protocol

- Targets: D2R, Mpro, TYK2, and USP7.
- Paired seeds: 0 through 9.
- Initial measured set: 60 compounds generated from the target and seed exactly as in the completed benchmark.
- Acquisition: 10 cycles of 30 compounds, ending at 360 measured compounds.
- Surrogate: exact Gaussian process with the existing fixed Tanimoto kernel and ECFP radius 4, 4,096-bit features.
- Controller: GPT-5.6 Sol Codex tasks for the no-API study.
- Action: nonnegative integer allocations to exploit, uncertain, and diverse queues summing to 30.
- Candidate ranking, overlap removal, and tie-breaking remain deterministic.
- Primary endpoint: top-2% recall at 360 measurements.
- Secondary endpoints: top-1% and top-5% recall, simple regret, unique top-5% scaffolds, and area under the top-2% recall curve.
- Statistical unit: one complete target/seed campaign. Cycles are not treated as independent replicates.
- Comparisons: paired target/seed differences with campaign-block bootstrap confidence intervals and per-target reporting.
- Stop conditions: hidden-label exposure, a changed frozen configuration, more than 1% invalid actions after one debugging pass, incorrect budgets, duplicate selections, or mismatched initial-set hashes.

## Information study

The formal study has five contemporaneous arms so that each ablation is compared with a reference generated under the same logging and forecast schema.

1. `full_reference`: GP mean and uncertainty summaries, chemical-space summary, revealed campaign history, and observed-only fingerprint-enrichment evidence.
2. `minus_uncertainty`: omit GP uncertainty summaries and uncertainty-derived history fields. The uncertainty queue remains available so only information, not mechanics, changes.
3. `minus_diversity`: omit chemical-space and diversity-derived history fields. The diversity queue remains available.
4. `minus_memory`: omit explicit prior decisions and outcomes from the agent view. The current GP and observed-affinity summary necessarily retain information from prior assays.
5. `shuffled_evidence`: retain the evidence scores, signs, ranks, and volume from `full_reference`, but apply a deterministic non-identity permutation to fingerprint-bit identities within the current observed-only evidence packet.

Explanatory evidence is computed only from already observed compounds. Its public function must not accept hidden pool affinities, top-active flags, or the full label vector. Shuffle randomness uses a SHA-256-derived seed that cannot alter campaign initialization.

The full matrix is 200 campaigns and 2,000 controller decisions. Before expansion, the development gate is one target (D2R), two seeds, and all five arms: 10 campaigns and 100 decisions. Expansion occurs only after all schema, leakage, action, budget, and initial-hash checks pass.

## Forecast calibration and rationale scoring

Legacy scoring is kept separate from the prospective study.

### Legacy audit

- Reconstruct batch affinity mean, batch affinity maximum, incumbent improvement, and newly observed scaffolds for all 400 completed decisions.
- Parse only explicit numerical threshold clauses in `expected_outcome`.
- Score each atomic clause as true or false and report conjunction success, fraction of clauses satisfied, signed threshold margin, and the unscorable rate.
- Vague terms without a defined threshold are unscorable, not false.
- Do not call the legacy hit rate probabilistic calibration because the old schema contains no probabilities.

### Prospective forecasts

Each new action must include:

- q10, q50, and q90 forecasts for next-batch affinity mean, affinity maximum, and new-scaffold count;
- a probability for whether the incumbent best will improve;
- up to three optional structured threshold claims;
- structured rationale assertions that reference fields in the saved pre-decision state.

Numeric forecasts are scored with pinball loss, median absolute error, quantile coverage, and 80% interval coverage/width. Binary forecasts are scored with Brier score, log loss, event rate, sharpness, and five equal-count reliability bins. Rationale reporting covers assertion grounding, evidence completeness, forecast linkage, and declared-priority/allocation consistency. Bootstrap resampling uses whole target/seed campaign blocks.

## Availability-constrained study

- Constraint type: deterministic 80% library availability mask derived from SHA-256 of canonical SMILES and a frozen protocol salt.
- The original 60-compound initial set is grandfathered as available so constrained and unconstrained campaigns remain paired at initialization.
- Unavailable compounds are removed from every post-initial candidate pool before GP prediction summaries and queue resolution.
- The same target-level mask is used for every method and seed; the mask generator never reads affinity labels.
- Run the constrained GPT-5.6 Sol controller for four targets and ten seeds (40 campaigns), plus the nine deterministic fixed policies under the identical mask (360 campaigns).
- Compare constrained versus unconstrained performance within target/seed/method and report the fraction of the library excluded and any endpoint-budget failures.

## Versioned API reproducibility check

The check is deliberately small: D2R seeds 0 and 1 under the frozen full-reference prompt, 20 decisions total. It records the exact model snapshot returned by the API, request IDs, token counts, latency, retry status, and serialized inputs/outputs. It is reported separately from Codex-task campaigns.

Execution is currently blocked because this host has neither `OPENAI_API_KEY` nor the OpenAI Python package. No API calls will be attempted and no dependency will be installed until credentials are provided and paid-call authorization is explicit.

## Implementation and execution order

1. Add config-driven information variants, observed-only explanatory evidence, deterministic shuffling, structured forecast schema, canonical decision logging, and availability filtering.
2. Add tests for exact view keys, no hidden-label parameters, shuffle invariants, initial-set hashes, selection uniqueness, budgets, forecast validation, and availability masks.
3. Backfill and score the 400 legacy forecasts; write claim-level, decision-level, and summary CSV outputs.
4. Run the 10-campaign information development gate and validate every trace.
5. Expand the five-arm information matrix to 200 campaigns if the gate passes.
6. Run constrained fixed policies, then the 40 constrained controller campaigns.
7. Run the 20-call versioned API check only after the credential/cost gate is cleared.
8. Aggregate paired statistics, generate figures, update the technical report, render it, and inspect every page.

## Planned artifacts

- Frozen JSON configurations and JSON schemas.
- Resumable per-campaign sessions with config, prompt, view, and evidence digests.
- Legacy and prospective calibration CSVs with validation JSON.
- Information-ablation and availability-constrained trajectory/decision outputs.
- Updated palette-matched figures and technical report.

