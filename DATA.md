# Data Provenance and Reproduction Guide

This document maps every read/write dependency in the pipeline, states what
must be downloaded before anything will run, gives the exact execution order,
and flags every place a methodological choice was made that isn't obvious
from the code alone.

## 1. What you need to download first

### VitalDB (accessed live via the Python API, no manual download)
`notebooks/01`, `08`, `09`, `10`, and `notebooks/eda/waveform_viewer.ipynb` call
the `vitaldb` package directly (`vitaldb.load_case`, `vitaldb.get_track_names`,
and the `https://api.vitaldb.net/cases` / `/labs` REST endpoints). No local
file is needed for this — just internet access. VitalDB is an open research
dataset; no credentialing was required for the tracks and case/lab data used
here at the time this pipeline was built.

### PhysioNet - VitalDB Arrhythmia Database (manual download required)
Several notebooks read per-patient beat/rhythm annotation files
(`Annotation_file_<caseid>.csv`) from the **VitalDB Arrhythmia Database**
("An anesthesiologist-validated, large-scale intraoperative arrhythmia
dataset with beat and rhythm labels", PhysioNet, v1.0.0).

**Download it, then place it so this exact path exists relative to the repo root:**

```
external_data/vitaldb-arrhythmia-database-1.0.0/Annotation_Files/Annotation_file_<caseid>.csv
```

`external_data/` is gitignored - it is never committed to this repo. One
notebook (`02_verify_and_add_rhythm_labels.ipynb`) also downloads
`metadata.csv` directly from PhysioNet over HTTPS (no manual step needed for
that file specifically).

**Check the PhysioNet license/terms for this database before redistributing
anything derived from it beyond what's already committed here** — this repo
ships aggregated, de-identified derived CSVs (episode counts, rhythm labels,
summary statistics), not the raw annotation files themselves, but you should
confirm this is compliant with the database's terms for your use case.

## 2. Full dependency chain (execution order)

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 01 | `merge_clinical_lab_data.ipynb` | annotation filenames (local, for case ID list) + VitalDB API (`/cases`, `/labs`) | `data/interim/merged_482_cases.csv` |
| 02 | `verify_and_add_rhythm_labels.ipynb` | `merged_482_cases.csv` + PhysioNet `metadata.csv` (HTTPS) + annotation files | `merged_482_cases.csv` (adds `dominant_rhythm`), `data/interim/rhythm_filtered_cases.csv` |
| 03 | `handle_missing_values.ipynb` | `rhythm_filtered_cases.csv` | `data/interim/imputed_477_cases.csv` |
| 04 | `extract_arrhythmia_features.ipynb` | `imputed_477_cases.csv` + annotation files | `data/interim/arrhythmia_features_draft.csv` |
| 05 | `verify_arrhythmia_features.ipynb` | `arrhythmia_features_draft.csv` + `imputed_477_cases.csv` | `data/interim/arrhythmia_features_482.csv` (first pass) |
| 06 | `extract_arrhythmia_episodes.ipynb` | `imputed_477_cases.csv`, `arrhythmia_features_482.csv` + annotation files | `data/interim/arrhythmia_episodes_draft.csv`, `data/interim/arrhythmia_features_updated_draft.csv` |
| 07 | `verify_arrhythmia_episodes.ipynb` | both draft files + `imputed_477_cases.csv` | `data/interim/arrhythmia_episodes_482.csv`, `arrhythmia_features_482.csv` (final, overwrites 05's version) |
| 08 | `nibp_bp_source_labeling.ipynb` | `arrhythmia_episodes_482.csv`, `imputed_477_cases.csv` + VitalDB API (track names, NIBP waveforms) | `data/processed/arrhythmia_episodes_updated.csv`, `clinical_lab_updated.csv`, `arrhythmia_episodes_summary.csv`, `clinical_lab_summary.csv` |
| 09 | `ioh_labeling.ipynb` | `arrhythmia_episodes_updated.csv` + VitalDB API (ART_MBP / NIBP waveforms) | `data/processed/episode_hypotension_labels.csv` |
| 10 | `merge_model_dataset.ipynb` | `arrhythmia_episodes_updated.csv`, `clinical_lab_updated.csv`, `episode_hypotension_labels.csv` + VitalDB API (HR/SpO2 waveforms) | `data/final/model_dataset.csv` |
| 11 | `build_ablation_datasets.ipynb` | `model_dataset.csv` | `data/final/no_map_dataset.csv`, `no_rhythm_dataset.csv`, `no_map_no_rhythm_dataset.csv` |
| 12 | `build_rhythm_only_dataset.ipynb` | `model_dataset.csv` + annotation files | `data/final/rhythm_only_dataset.csv` |
| — | `eda/clinical_histograms.ipynb` | `data/processed/clinical_lab_updated.csv` | `figures/clinical_hist_*.png` |
| — | `eda/correlation_heatmap.ipynb` | `data/final/model_dataset.csv` | `figures/correlation_heatmap.png` |
| — | `eda/missing_data_plots.ipynb` | `data/final/model_dataset.csv` | `figures/missingness_table.{csv,png}`, `figures/missing_matrix.png` |
| — | `eda/waveform_viewer.ipynb` | `data/processed/arrhythmia_episodes_updated.csv` + annotation files + live VitalDB waveforms | ad-hoc PNG per run (interactive QA tool, not part of the pipeline) |
| — | `ml_pipeline` (`python -m ml_pipeline.main --csv data/final/<dataset>.csv`) | one of the 5 `data/final/*.csv` | `outputs/<dataset>/*` (results table, ROC/boxplot figures, SHAP plot, model_results.csv) |
| — | `analysis/*.py` (3 scripts) | `data/interim/imputed_477_cases.csv`, `data/interim/arrhythmia_episodes_482.csv` + VitalDB API | printed console output only (informed the ART_MBP/NIBP methodology decision — not a pipeline stage) |

`archive/rhythm_extraction.ipynb` and `archive/view_file.py` are excluded from
this chain entirely — they are scratch files kept for history, not pipeline
stages (see the repo's file classification for why).

## 3. Root inputs (produced by nothing in this repo)

| Input | Source |
|---|---|
| `external_data/.../Annotation_Files/*.csv` | PhysioNet (VitalDB Arrhythmia Database v1.0.0) — manual download, see §1 |
| Case/lab clinical data | VitalDB Python API (`vitaldb.get_case_info` / `/cases`, `/labs` REST endpoints) — live, no download |
| Waveform tracks (ART_MBP, NIBP_SBP/DBP, HR, SpO2, ECG) | VitalDB Python API (`vitaldb.load_case`) — live, no download |
| PhysioNet `metadata.csv` | Downloaded automatically by notebook 02 over HTTPS — no manual step |

## 4. Blockers — resolved

An earlier audit of this pipeline found 3 gaps where a committed CSV had no
notebook in the repo that could reproduce it. All 3 are now closed:

1. **`rhythm_filtered_cases.csv`** had no producing notebook. **Fixed**: added
   as Part E of `02_verify_and_add_rhythm_labels.ipynb` (drops the 5 patients
   whose `dominant_rhythm == "Noise"`; no patient in this cohort is
   `"Unclassifiable"`).
2. **`no_map_dataset.csv`, `no_rhythm_dataset.csv`, `no_map_no_rhythm_dataset.csv`**
   had no producing notebook — they were originally built with one-off scripts
   that were never committed. **Fixed**: new `11_build_ablation_datasets.ipynb`,
   verified to reproduce the existing files exactly (to float64 precision —
   the only differences found were ~1e-31 magnitude noise on near-zero slope
   features, i.e. floating-point representation noise, not a real
   discrepancy).
3. **The label join into `model_dataset.csv`** was undocumented: the notebook
   said the join happens "at training time" externally, but the actual
   committed file already had labels joined in, with 88 no-data episodes
   recoded to `hypotension_label=0` rather than dropped as documented.
   **Fixed**: the join is now an explicit final cell in
   `10_merge_model_dataset.ipynb`, with the rationale below.

**Note on the label-join methodology** (documented here since it's a real,
debatable choice affecting your reported class balance): 88 of 1,284 episodes
(6.9%) have `outcome_window_availability` near 0% — essentially no BP
monitoring in the 5 minutes after the episode, so there's no way to confirm
whether hypotension occurred. This pipeline recodes those as
`hypotension_label = 0` (no confirmed complication) rather than excluding
them, on the reasoning that inconclusive monitoring isn't evidence of an
adverse outcome. The alternative — dropping them — would shrink
`model_dataset.csv` from 1,284 to 1,196 rows and change the reported negative
count from 838 to 750. This pipeline's committed results use the recode-to-0
version; if you change this, every downstream dataset variant and all
`ml_pipeline` outputs need to be regenerated.

No other blockers were found. Every other file in `data/` is reproducible by
running the notebooks in the order in §2.

## 5. Regenerate vs. commit

**Fully regenerable** by re-running the notebook chain (given the PhysioNet
download in place and VitalDB API access): every file in `data/interim/`,
`data/processed/`, and `data/final/`.

**Committed anyway**, despite being regenerable, because full regeneration of
the *data* needs internet access, PhysioNet credentialing/download, and
roughly an hour of VitalDB API calls (§6) — a reviewer who just wants to check
the modeling results shouldn't have to wait for that:
- All of `data/interim/`, `data/processed/`, `data/final/` (17 CSVs, all
  under 1.2 MB — see the repo's file inventory for exact sizes)
- `figures/*.png` (correlation heatmap, missingness table, dataset comparison
  table, ECG case figures) — these are fixed, one-off analysis outputs, not
  something regenerated on every `ml_pipeline` run

**Not committed** (gitignored): the *generated content* inside `outputs/`
(`*.png` and `*.csv` in each subfolder — ROC curves, AUROC boxplot,
confusion matrix, results table, SHAP plot, `model_results.csv`). The 5
subfolders themselves (one per dataset in `data/final/`) and their
`README.md`/`.gitkeep` placeholders ARE committed, so the structure is
visible on GitHub even though each folder starts empty. Unlike the raw data
above, these run artifacts are cheap and fast to regenerate (seconds to low
minutes per dataset, no internet or PhysioNet access needed — training runs
entirely on the already-committed `data/final/*.csv` files), so the repo
ships without pre-generated figures on purpose — see `outputs/README.md`.
Run:

```
python -m ml_pipeline.main --csv data/final/<name>.csv --output_dir outputs/<name>
```

for each of the 5 datasets in `data/final/` to populate that folder with your
own trained models and figures. `main.py` creates the output directory
automatically and clears any model-named files from a prior run before
writing new ones, so re-running never leaves a stale figure from a different
winning model behind.

Also not committed: `external_data/` (the PhysioNet download itself —
redistribution terms unconfirmed, see §1) and anything under
`.venv/`/`__pycache__/`.

## 6. How to run it, and how long it takes

1. `pip install -r requirements-notebooks.txt` (data-prep environment)
2. Download the PhysioNet database and place it per §1
3. Run `notebooks/01` through `notebooks/12` in numeric order (the `eda/`
   subfolder notebooks can run any time after their inputs exist — they're
   not on the critical path)
4. `pip install -r requirements.txt` (separate environment for the training
   pipeline — the two environments pin different library versions on purpose,
   see the repo's environment notes)
5. `python -m ml_pipeline.main --csv data/final/<dataset>.csv --output_dir outputs/<dataset>`
   for each of the 5 datasets in `data/final/`

**Approximate timing** (dominated by VitalDB API calls, varies with network
conditions — these are rough, not measured benchmarks):

| Stage | Rough time |
|---|---|
| Notebooks 01–07 (annotation-file scans, no waveform loading) | 10–15 min combined |
| Notebook 08 (NIBP source labeling — waveform loads for 477 patients) | 5–10 min |
| Notebook 09 (IOH labeling — waveform loads for 457 patients) | 15–25 min |
| Notebook 10 (model dataset merge — HR/SpO2 waveform extraction) | 10–20 min (the notebook's own comment flags this as the slow step) |
| Notebooks 11–12 | under 1 min combined |
| `eda/` notebooks | under 1 min each, except `waveform_viewer.ipynb` (seconds per case, interactive) |
| `ml_pipeline` full run, per dataset (4 models, `RandomizedSearchCV`, 1000-resample bootstrap, SHAP) | 2–10 min per dataset |

**Full pipeline, start to finish, all 5 datasets trained: roughly 1–1.5 hours**,
most of it waiting on VitalDB API responses rather than compute.
