# Agentic Active Learning Results Explained Step by Step

## 1. What question the experiment asks

The experiment asks whether a language-model controller can decide how to spend each new assay batch during ligand discovery. The controller does not predict binding affinity itself and does not see the hidden answers. A Gaussian process predicts affinity and uncertainty. The controller only divides each batch of 30 compounds among three queues:

- **Exploit:** choose compounds with the highest predicted affinity.
- **Uncertain:** choose compounds where the model is least certain.
- **Diverse:** choose compounds that broaden chemical coverage.

The practical question is whether changing this mixture from cycle to cycle works better than random selection and whether it can match a well-chosen fixed mixture.

## 2. What was run

The core benchmark used four targets: D2R, Mpro, TYK2, and USP7. Each campaign started with 60 measured compounds, added ten batches of 30, and ended at 360 measurements. Ten paired random seeds were used per target. Pairing means that, for a target and seed, every method began from exactly the same 60 compounds.

The completed experiment set contains:

- 40 core agent campaigns and 400 agent decisions.
- 360 core fixed-policy campaigns covering nine policies.
- 200 information-ablation campaigns.
- 40 availability-constrained agent campaigns.
- 360 availability-constrained fixed-policy campaigns.

## 3. How to read the main metric

The primary endpoint is **top-2% recall at 360 measurements**. It asks: among all compounds that truly belong to the best 2% of the full library, what fraction did the campaign discover?

For example, D2R recall of 0.412 means that the controller found 41.2% of the library's true top-2% compounds. It does not mean that 41.2% of the measured compounds were hits.

Top-1% recall is stricter and emphasizes the very best compounds. Top-5% recall is broader. Simple regret is the gap between the best discovered affinity and the true best affinity; zero means the campaign found the global best compound. The scaffold count measures chemical breadth among discovered top-5% compounds.

## 4. Core endpoint result

![Core endpoint comparison](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/endpoint_top2_comparison.png)

| Target | Agent | Random | Strongest fixed policy | Agent minus strongest fixed |
|---|---:|---:|---:|---:|
| D2R | 0.412 | 0.146 | 0.494 balanced | -0.082 |
| Mpro | 0.374 | 0.120 | 0.370 explore-heavy | +0.004 |
| TYK2 | 0.448 | 0.036 | 0.519 exploit | -0.071 |
| USP7 | 0.683 | 0.180 | 0.694 explore-heavy | -0.011 |

The first conclusion is strong: the controller beat random selection in all 40 matched campaigns. It therefore learned useful information from the surrogate summaries.

The second conclusion is more qualified. The controller did not universally beat the best fixed policy. It was statistically tied with the strongest fixed policy on Mpro and USP7, and it was worse on D2R and TYK2. The paper should claim **consistent improvement over random and target-dependent competitiveness**, not universal superiority.

## 5. What happened on each target

### D2R

The agent recovered 41.2% of the top-2% set. Random recovered 14.6%, so the controller added 26.6 percentage points. The balanced fixed policy reached 49.4%, leaving the controller 8.2 points behind. The 95% paired bootstrap interval for the agent-minus-balanced difference was -15.0 to -2.0 points, so the deficit is unlikely to be explained by seed noise alone. The controller found the single best compound in seven of ten campaigns.

Plain meaning: D2R benefited from a stable balanced recipe more than from the controller's changing recipe.

### Mpro

The agent recovered 37.4% of the top-2% set, versus 12.0% for random and 37.0% for the explore-heavy fixed policy. The +0.4-point difference from the strongest fixed policy had a confidence interval that crossed zero. All ten agent campaigns found the global best compound.

Plain meaning: on Mpro, the controller independently arrived at behavior that was about as effective as the best hand-designed policy.

### TYK2

The agent recovered 44.8% of the top-2% set, compared with only 3.6% for random. Pure exploitation reached 51.9%. The controller's paired deficit was 7.1 points, with a 95% interval from -12.5 to -2.0 points. Every agent campaign still found the global best compound.

Plain meaning: TYK2 had a strong enough predictive ranking that spending almost everything on exploitation worked best. The controller adapted in that direction, but not aggressively or early enough to match the fixed exploit policy.

### USP7

USP7 gave the strongest absolute result. The agent recovered 68.3% of the top-2% set and 60.4% of the top-5% set. The best fixed policy reached 69.4%, only 1.1 points higher, and the confidence interval crossed zero. Every agent campaign found the global best compound.

Plain meaning: the controller was highly effective on USP7 and essentially tied the best fixed recipe.

## 6. Recall at different activity thresholds

![Recall levels](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/endpoint_recall_levels.png)

| Target | Top-1% recall | Top-2% recall | Top-5% recall | Mean simple regret | Mean top-5% scaffolds |
|---|---:|---:|---:|---:|---:|
| D2R | 0.481 | 0.412 | 0.332 | 0.045 | 25.5 |
| Mpro | 0.468 | 0.374 | 0.289 | 0.000 | 33.4 |
| TYK2 | 0.588 | 0.448 | 0.284 | 0.000 | 21.4 |
| USP7 | 0.800 | 0.683 | 0.604 | 0.000 | 29.8 |

D2R is the only target with nonzero mean regret because three runs missed the global optimum. Mpro, TYK2, and USP7 have zero mean regret because every run found the single best compound. TYK2 combines strong top-1% recall with much lower top-5% recall; this indicates concentrated success near the extreme top rather than broad recovery across the larger high-affinity set.

## 7. When the discoveries happened

![Learning curves](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/top2_learning_curves.png)

The learning curves plot recall after the initial 60 compounds and after each additional batch. A curve that rises earlier is more useful operationally because it finds good compounds before the full budget is spent.

The agent separated from random early on every target and kept that advantage. Mpro and USP7 stayed close to their strongest fixed policies for much of the campaign. D2R and TYK2 developed persistent gaps in favor of the target-specific fixed policy. This means the endpoint differences were not caused by only one lucky final batch.

## 8. How the controller allocated each batch

![Batch allocations](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/agent_allocations_by_cycle.png)

Across all targets, the mean exploit allocation rose from 8.2 compounds in cycle 1 to 18.6 in cycle 10. Over the same period, uncertainty fell from 11.6 to 6.7 and diversity fell from 10.2 to 4.7. The controller therefore used early batches to learn and later batches to harvest.

The target averages were:

| Target | Exploit | Uncertain | Diverse | Interpretation |
|---|---:|---:|---:|---|
| D2R | 9.09 | 11.76 | 9.15 | Broad early search followed by late exploitation |
| Mpro | 5.80 | 12.90 | 11.30 | Persistent investment in learning and coverage |
| TYK2 | 19.05 | 7.22 | 3.73 | Strong exploitation |
| USP7 | 12.44 | 10.19 | 7.37 | Mixed early strategy with stronger late exploitation |

This proves that the controller's actions changed with target and campaign state. It does not by itself prove that the written rationale faithfully reveals why the model made each choice.

## 9. Comparison with all fixed policies

![Agent versus every fixed policy](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/agent_vs_policy_heatmap.png)

Each heat-map cell is agent recall minus fixed-policy recall. Positive values favor the agent; negative values favor the fixed policy. Figure 5 uses the requested Phthalo Blue scale ending at `#0E155E`, while the signed numbers preserve the direction of every comparison.

The agent beat many weak or mismatched fixed policies by a large margin, especially random and pure uncertainty. Negative cells cluster around policies that match a target unusually well, such as balanced on D2R and exploit on TYK2. The average column is useful as a summary but should never replace the per-target cells.

## 10. Forecast testability and success

![Forecast scoring](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/forecast_testability_and_success.png)

Before each batch, the controller wrote an expected outcome. The scorer looked for explicit numerical threshold claims and compared them with the next batch.

- 321 of 400 forecasts were scorable.
- 79 forecasts, or 19.75%, were too qualitative for the rule-based scorer.
- The scorer extracted 686 atomic threshold claims.
- 86.30% of atomic claims succeeded.
- On average, 87.28% of clauses in a scorable forecast succeeded.
- Every clause succeeded together in 71.65% of scorable forecasts.

D2R and TYK2 forecasts were always scorable. Mpro was scorable 83% of the time. USP7 was scorable only 38% of the time because its wording was often qualitative. Among scorable forecasts, conjunction success ranged from 65.0% on TYK2 to 76.3% on USP7.

This is a **testability and hit-rate analysis**, not formal probability calibration. The forecasts did not include probabilities, so Brier scores, reliability diagrams, and statements such as “80% confidence events occurred 80% of the time” cannot be reconstructed.

## 11. Information ablations

![Information ablations](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/information_ablation_endpoints.png)

Each ablation removed one information channel while keeping the target, seed, budget, and other protocol elements paired.

| Target | Full rerun | No uncertainty | No diversity | No memory | Shuffled evidence |
|---|---:|---:|---:|---:|---:|
| D2R | 0.488 | 0.504 | 0.472 | 0.484 | 0.488 |
| Mpro | 0.381 | 0.383 | 0.383 | 0.385 | 0.381 |
| TYK2 | 0.344 | 0.347 | 0.344 | 0.340 | 0.344 |
| USP7 | 0.640 | 0.626 | 0.640 | 0.637 | 0.640 |

Removing uncertainty had little effect except a 1.4-point loss on USP7. Removing diversity caused a 1.6-point loss on D2R and essentially no loss elsewhere. Removing memory changed means by less than half a percentage point. Shuffling explanatory evidence changed the mean by exactly zero on every target.

The decision paths reinforce the shuffled-evidence result: shuffled and full allocations were identical for all ten D2R, Mpro, and TYK2 seeds and nine of ten USP7 seeds. This suggests that fingerprint-bit identities in the evidence packet usually did not drive the controller's allocation.

The full-reference values in this table differ from the core benchmark because the controller ran through Codex tasks rather than a frozen deterministic model snapshot. These rows must be compared only with their paired ablation arms. They must not be treated as a failed reproduction of the core means.

## 12. Availability constraint

![Availability comparison](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/availability_agent_comparison.png)

The operational benchmark made a deterministic 80% of each library available after the shared initial set. The mask used canonical isomeric SMILES and a salted SHA-256 rule, without using affinity labels.

| Target | Full agent | 80% availability agent | Change | Best fixed at 80% |
|---|---:|---:|---:|---:|
| D2R | 0.412 | 0.342 | -0.070 | 0.422 balanced |
| Mpro | 0.374 | 0.302 | -0.072 | 0.293 exploit-heavy |
| TYK2 | 0.448 | 0.334 | -0.114 | 0.472 exploit |
| USP7 | 0.683 | 0.614 | -0.069 | 0.577 explore-heavy |

The restriction reduced agent recall on every target, most strongly on TYK2. Even under the constraint, the agent beat the strongest constrained fixed policy on Mpro and USP7, while fixed policies remained ahead on D2R and TYK2. All 12,000 post-initial agent selections were available, so the result is not contaminated by invalid choices.

![Fixed policies under availability](/Users/vaidehisaxena/Documents/ChatGPT/Disha_neurips/figures/technical_report/availability_policy_comparison.png)

The fixed-policy comparison shows that availability also lowered the best fixed baselines. The leading constrained policy lost 7.2 points on D2R, 5.2 on Mpro, 4.7 on TYK2, and 11.7 on USP7 relative to the same unconstrained policy. Candidate access is therefore a property of the problem, not merely a weakness of the controller.

## 13. Validation and what can be trusted

The Zenodo files matched all published byte sizes and MD5 checksums, and the archive integrity test passed. The four target tables had unique nonempty SMILES and finite affinities. Campaign budgets, cycle counts, initial-set pairing, and selection uniqueness were validated. The test suite passed 40 tests.

The ablation validation status is `PASS_WITH_WARNING`, not a failure. Ten early D2R gate sessions omitted a redundant top-level initial-set digest, but their derived digests and paired initial sets were reconstructed and matched. No incomplete campaign entered the reported analyses.

## 14. What the evidence supports

The strongest defensible conclusion is:

> A GPT-5.6 Sol controller can adapt the exploit-uncertain-diverse allocation over a ligand-discovery campaign, consistently outperform random sampling, and approach the strongest fixed policy on some targets. Performance remains target-dependent, operational availability lowers attainable recall, and most supplied explanatory evidence is not yet shown to causally influence the controller.

The evidence does not support saying that the agent is universally best, that written rationales expose the true internal cause of decisions, or that the forecasts are probability-calibrated.
