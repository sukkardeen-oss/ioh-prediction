# Outputs

This directory is intentionally empty in the repository — it holds
`ml_pipeline` run artifacts, which are cheap to regenerate locally and
therefore not committed (see `DATA.md` §5).

Each subfolder corresponds to one dataset variant in `data/final/`. To
populate a subfolder with your own trained models and figures, run from the
repo root:

```
python -m ml_pipeline.main --csv data/final/<name>.csv --output_dir outputs/<name>
```

replacing `<name>` with one of: `model_dataset`, `no_map_dataset`,
`no_rhythm_dataset`, `no_map_no_rhythm_dataset`, `rhythm_only_dataset`.

Each run produces: `model_results.csv`, `roc_curves_combined.png`,
`auroc_comparison_boxplot.png`, `confusion_matrix_<best_model>.png`,
`results_table.png`, `shap_summary_<best_model>.png`,
`interpretability_summary.csv`. `main.py` clears any stale model-named files
from a previous run before writing new ones, so re-running never mixes
figures from two different winning models.
