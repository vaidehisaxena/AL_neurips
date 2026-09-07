# Source provenance

Verified on 2026-09-07.

## Upstream code

- Repository: <https://github.com/meyresearch/explainable_AL>
- Imported branch: `main`
- Imported commit: `c77e3a6a77e0207f995815c22f3146ae9b046777`
- License: MIT (`LICENSE` in the imported repository)
- Local remote: `upstream`

The upstream Git history was imported before AgenticAL changes. The first local project commit is `d05596e` (`Add AgenticAL experiment plan and artifact safeguards`).

## Zenodo artifacts

- Record: <https://zenodo.org/records/17935028>
- DOI: `10.5281/zenodo.17935028`
- Title: *Explainable Active Learning Framework for Ligand Binding Affinity Prediction*
- Local directory: `artifacts/zenodo/17935028/` (ignored by Git)
- Raw record metadata: `artifacts/zenodo/17935028/zenodo-record.json`

| File | Bytes | Published MD5 | Verification |
|---|---:|---|---|
| `composite_d2r.csv` | 31,556,688 | `8711364364949eb1b24c6f13ea56e527` | match |
| `mpro_sorted.csv` | 1,054,777 | `cdbd667a1b3df6808110b86f7aad4263` | match |
| `mw_comparison_metrics.csv` | 104,052 | `ba76f159f82fb82b8e1fa60bf8c373b4` | match |
| `TYK2_sorted.csv` | 778,943 | `8d87942c7c7457cb476c5c9e7f0e02df` | match |
| `USP7_sorted.csv` | 2,169,120 | `80eda5b5c5da5df39235879d2b3a621e` | match |
| `recall_combined_results (3).csv` | 1,168,984 | `ddb211aead98622936a538a35ad5b473` | match |
| `Explainable AL Dataset_Result data.zip` | 442,365,836 | `03b6d85c0f12f0a8fb17c9748a3d3ad9` | match; `unzip -t` passed |

The large archive was retrieved through byte-range requests after the single-connection Zenodo endpoint slowed severely. The ranges were assembled to a separate file; the incomplete initial transfer was replaced only after byte count and MD5 matched the Zenodo metadata. Temporary range files were then deleted.

### Initial schema audit

The four target CSVs contain 2,502 D2R, 2,749 Mpro, 9,997 TYK2, and 1,799 USP7 data rows. All have nonempty, unique SMILES and finite parseable affinities, and their target columns are internally consistent. `TYK2_sorted.csv` has three additional blank header fields caused by trailing delimiters; those fields are empty in every row. This should be normalized explicitly in the future data-loading layer, not by modifying the raw artifact.
