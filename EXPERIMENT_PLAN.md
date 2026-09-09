# AgenticAL: implementation and experiment plan

Status: **data acquisition, readiness checks, and the fixed-policy benchmark are complete. Paid agent experiments remain behind the cost/API-key gate in Phase 5.**

## 1. Objective and core hypothesis

Build a leakage-safe virtual closed-loop ligand-discovery benchmark on top of the published `explainable_AL` codebase and data. Test whether a single tool-using discovery agent can adaptively allocate each assay batch between exploitation, uncertainty-driven exploration, and chemical-space diversification better than policies fixed before a campaign begins.

Primary hypothesis: an agent observing the evolving campaign can infer useful properties of the target-specific chemical landscape and outperform strong fixed or hand-coded adaptive policies on discovery efficiency.

The LLM will not replace the Gaussian process (GP), calculate affinities, or directly choose from raw unrestricted SMILES. Numerical tools rank candidates deterministically; the agent performs strategy/meta-control under a strict budget.

## 2. Reproducibility contract

- Upstream code: `https://github.com/meyresearch/explainable_AL`
- Paper artifact record: Zenodo `10.5281/zenodo.17935028`
- Record exact upstream commit SHA, Zenodo metadata/version, file URLs, sizes, and checksums in a machine-readable provenance manifest.
- Preserve upstream history in the new repository and make all new work on a `codex/agentic-al` branch.
- Never commit secrets, API keys, virtual environments, caches, downloaded bulk artifacts, or generated experiment results.
- Pin the Python/runtime dependencies after reproducing the upstream test suite.
- Store every run's configuration, seed, code commit, environment fingerprint, prompts/model settings, decisions, tool outputs, and metrics.
- All comparisons use identical initial labeled sets per target/seed and identical total assay budgets.

## 3. Data acquisition and validation

1. Query the Zenodo API for record `17935028` and save its metadata.
2. Download every artifact into `artifacts/zenodo/17935028/` using resumable transfers.
3. Verify each file against the checksum and size published by Zenodo; retain a verification report.
4. Inventory archive contents without executing anything from the archive.
5. Validate the four target datasets (TYK2, USP7, D2R, Mpro): schema, row counts, missing values, duplicate IDs/SMILES, affinity direction and units, target labels, and top-active flags.
6. Compare overlapping datasets/artifacts with the files shipped in the upstream repository and document differences.
7. Enforce a hidden-label boundary: only the environment may access the complete affinity column; agent-facing state and tools receive labels solely after `run_assay` reveals them.

Failure policy: stop and report any inaccessible file, checksum mismatch, corrupt archive, ambiguous license, inconsistent affinity direction/unit, or unreconcilable mismatch with repository data. Do not silently substitute another dataset.

## 4. Repository setup

1. Clone `meyresearch/explainable_AL` into a temporary staging directory and inspect its default branch, commit, tags, license, dependency files, tests, and data layout.
2. Use the current empty local repository as the new project `AL_neurips`; import the upstream code and history without overwriting unrelated user work.
3. Keep an `upstream` remote pointing to `meyresearch/explainable_AL`; create an `origin` remote for the user's new GitHub repository `AL_neurips`.
4. Create the GitHub repository as **private by default** through the user's signed-in GitHub interface, then push `main` and the implementation branch.
5. Add provenance, artifact-ignore rules, configuration templates, tests, and this plan before implementation.

If the GitHub account/organization or repository visibility cannot be determined from the signed-in interface, pause before remote creation and request that choice. Never overwrite an existing remote repository named `AL_neurips`.

## 5. Readiness gate (must pass before experiments)

- Upstream tests pass in a clean environment, or every pre-existing failure is documented.
- One tiny deterministic smoke campaign completes using a toy/copied dataset and no LLM calls.
- Hidden-label leakage tests pass.
- Batch size, total budget, uniqueness, and pool-exhaustion invariants pass.
- All four datasets pass schema and affinity-direction checks.
- Fixed-policy outputs can be reproduced for at least one target/seed.
- A dry-run enumerates the complete experiment matrix, estimated number of GP fits, LLM calls, runtime, storage, and API cost.
- The dry-run report receives explicit user approval before paid LLM experiments or the full benchmark begins.

## 6. System design

### Environment state

At cycle `t`, expose only summarized/authorized views of:

- observed compounds and revealed affinities;
- GP mean and calibrated uncertainty over the unlabeled pool;
- campaign history and remaining budget;
- chemical-space coverage, similarity, and scaffold statistics;
- optional SHAP/SAR evidence derived only from observed labels.

### Deterministic tools

- campaign summary and previous-cycle outcomes;
- top predicted, most uncertain, and diverse candidate summaries;
- GP diagnostics and uncertainty calibration summaries;
- fragment enrichment and SHAP fragment summaries;
- compound inspection/comparison;
- acquisition allocation and `run_assay` with schema validation.

### Agent action

For each batch of 30, output a validated allocation `(n_exploit, n_uncertain, n_diverse)` summing to 30, plus a structured, falsifiable rationale and expected outcome. Candidate ranking, overlap resolution, and tie-breaking are deterministic and seed-controlled. Invalid actions are rejected and logged rather than repaired invisibly.

## 7. Frozen minimum viable benchmark

- Targets: TYK2, USP7, D2R, Mpro.
- Representation: ECFP4 (match upstream radius/bit conventions exactly).
- Surrogate/kernel: select the most robust upstream GP configuration only after reproducing its reported behavior; freeze it for all methods.
- Initial labeled batch: 60.
- Acquisition cycles: 10.
- Batch size: 30.
- Total measured per campaign: 360, subject to pool size.
- Seeds: 10 paired seeds per target and method.
- No representation/kernel mega-grid in the primary study.

## 8. Methods and ablations

Primary baselines:

1. Random.
2. Pure uncertainty/explore.
3. Pure mean/exploit.
4. Fixed balanced UCB.
5. Best fixed schedule from the original study, chosen without using new test outcomes.
6. Hand-coded adaptive heuristic with prespecified thresholds.

Agent variants:

7. LLM without quantitative tools.
8. Agent + GP summaries.
9. Agent + GP + chemical-space tools.
10. Full agent: GP + chemistry + true SHAP + campaign memory.

Faithfulness ablations:

- full agent minus SHAP;
- full agent with cycle-matched shuffled SHAP;
- full agent minus memory/history;
- information-removal sensitivity tests for uncertainty and diversity signals.

Run baselines first. Run agent variants only after the cost/readiness gate, starting with 2 development seeds and one target, then expand only if logs and metrics are valid.

## 9. Metrics and statistical analysis

Primary scientific outcomes:

- best observed affinity versus measured-compound budget;
- simple regret versus budget;
- top-1%, top-2%, and top-5% recall versus budget;
- experiments-to-first-top-1% hit;
- area under the learning/discovery curve.

Secondary outcomes:

- unique Bemis–Murcko scaffolds among discovered high-affinity compounds;
- chemical diversity/redundancy and chemical-space coverage;
- cumulative regret and GP predictive/calibration diagnostics.

Agent outcomes:

- valid tool/action rate and budget compliance;
- decision stability across repeated inference at fixed state;
- grounding of rationale claims against tool outputs;
- sensitivity to removed or shuffled evidence;
- latency, token use, and monetary cost.

Use paired target/seed comparisons at matched budgets, report effect sizes and bootstrap confidence intervals, and avoid treating cycles from one campaign as independent replicates. Declare the primary endpoint and tie-breaking rule before viewing full agent results. Report per-target results as well as aggregate ranks because target landscape is central to the hypothesis.

## 10. Implementation sequence

1. Acquire and verify code/data; produce provenance and inventory reports.
2. Reproduce upstream tests and one published baseline configuration.
3. Refactor a minimal environment boundary around the existing GP/acquisition runner.
4. Add candidate-selection tools, structured schemas, constraints, logs, and leakage tests.
5. Implement fixed and hand-coded baselines; run the baseline matrix.
6. Add the single-agent controller and deterministic fallback/error handling.
7. Run smoke/development agent campaigns and audit every trace.
8. Freeze prompt, tool schemas, model/version, temperature, retry policy, and analysis plan.
9. Run the full paired benchmark in resumable jobs with per-run locks and immutable configs.
10. Aggregate metrics, confidence intervals, behavior audits, and the four planned figures.

## 11. Stop conditions

Stop execution and notify the user if:

- dataset provenance/integrity cannot be verified;
- hidden labels appear in prompts, tool outputs, caches, or agent logs before assay;
- upstream behavior cannot be reproduced within documented tolerances;
- the expected LLM cost or total compute exceeds the approved dry-run estimate;
- more than 1% of actions are invalid after one debugging pass;
- a run configuration changes after benchmark freeze;
- an existing GitHub repository or local nontrivial work would be overwritten.

## 12. Expected outputs

- `provenance/`: source commit, Zenodo metadata, checksums, licenses, environment lock.
- `configs/`: immutable campaign/method/model configurations.
- `src/` or upstream package extensions: environment, tools, policies, agent controller.
- `tests/`: leakage, invariants, determinism, tool schemas, metric correctness.
- `runs/`: ignored raw outputs with a manifest and resumable status.
- `results/`: tidy aggregate tables and statistical summaries.
- `figures/`: architecture, discovery curves, target-specific policy heatmaps, ablations.
- `docs/`: experimental protocol, prompt/model card, limitations, and reproduction guide.
