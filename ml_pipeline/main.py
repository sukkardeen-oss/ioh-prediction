"""IOH prediction pipeline: load -> patient-level split -> train & tune
Logistic Regression, Ridge Regression, Random Forest, XGBoost
-> evaluate -> select best -> combined ROC + AUROC boxplot for all models
-> confusion matrix + SHAP + results table for the best model only.

Usage:
    python -m ml_pipeline.main
    python -m ml_pipeline.main --csv path/to/model_dataset.csv --target hypotension_label
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)

from . import config
from .data_utils import load_data, split_data
from .interpretability import explain_best_model
from .train import run_all_models, select_best_model

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CSV = os.path.join(_REPO_ROOT, "data", "final", "model_dataset.csv")

# Consistent colours for all four models across every figure.
_MODEL_COLORS = {
    "Logistic Regression": "#e41a1c",
    "Ridge Regression":    "#377eb8",
    "Random Forest":       "#4daf4a",
    "XGBoost":             "#984ea3",
}


def _dataset_name(csv_path: str) -> str:
    """'no_map_dataset.csv'  ->  'No Map Dataset'"""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return stem.replace("_", " ").title()


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save_combined_roc(fitted: dict, X_test, y_test, dataset_name: str,
                       output_dir: str) -> str:
    """One ROC figure with all model curves overlaid, updated every run."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, estimator in fitted.items():
        try:
            y_proba = estimator.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            color = _MODEL_COLORS.get(model_name, "#888888")
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{model_name}  (AUC = {roc_auc:.3f})")
        except Exception as exc:
            print(f"    [warn] ROC skipped for {model_name}: {exc}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name}")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_auroc_boxplot(bootstrap_aurocs: dict, dataset_name: str,
                        output_dir: str) -> str:
    """Box-and-whisker plot of bootstrapped AUROC for all models, updated every run."""
    names  = list(bootstrap_aurocs.keys())
    data   = [bootstrap_aurocs[n] for n in names]
    colors = [_MODEL_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bp = ax.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    all_vals = np.concatenate(data)
    ax.set_ylim(bottom=max(0.0, float(all_vals.min()) - 0.05), top=1.0)
    ax.set_ylabel("AUROC (1 000 bootstrap resamples)")
    ax.set_title(f"AUROC Bootstrap Distribution — {dataset_name}")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "auroc_comparison_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_confusion_matrix(model_name: str, y_test, y_pred,
                            output_dir: str) -> str:
    """Confusion matrix — called for the best model only."""
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No IOH (0)", "IOH (1)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_results_table(results: pd.DataFrame, dataset_name: str,
                        output_dir: str) -> str | None:
    """Formatted PNG table of per-model metrics, updated every run."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return None

    def fmt_auroc(row):
        v  = row.get("roc_auc",         float("nan"))
        lo = row.get("roc_auc_ci_lower", float("nan"))
        hi = row.get("roc_auc_ci_upper", float("nan"))
        if pd.isna(v):
            return "—"
        return f"{v:.3f} ({lo:.3f}-{hi:.3f})"

    def fmt_pct(row, col):
        val = row.get(col, float("nan"))
        return "—" if pd.isna(val) else f"{val * 100:.1f}"

    metric_rows = [
        ("AUROC (95% CI)",           [fmt_auroc(r)              for _, r in ok.iterrows()]),
        ("Sensitivity, %\n(θ=0.5)", [fmt_pct(r, "sensitivity") for _, r in ok.iterrows()]),
        ("Specificity, %\n(θ=0.5)", [fmt_pct(r, "specificity") for _, r in ok.iterrows()]),
        ("PPV, % (θ=0.5)",          [fmt_pct(r, "ppv")         for _, r in ok.iterrows()]),
        ("NPV, % (θ=0.5)",          [fmt_pct(r, "npv")         for _, r in ok.iterrows()]),
    ]

    models     = ok["model"].tolist()
    col_labels = ["Metric"] + models
    cell_text  = [[label] + vals for label, vals in metric_rows]
    n_rows     = len(metric_rows)
    n_cols     = len(col_labels)

    fig, ax = plt.subplots(figsize=(2.5 + n_cols * 2.8, 1.4 + n_rows * 0.85))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.8)

    # Header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1e3d5f")
        cell.set_text_props(color="white", fontweight="bold")

    # Data rows — alternating shading, bold metric column
    for i in range(1, n_rows + 1):
        bg = "#eef2f7" if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            if j == 0:
                cell.set_text_props(fontweight="bold")

    ax.set_title(
        f"Model Performance Summary  |  Dataset: {dataset_name}",
        fontsize=11, fontweight="bold", pad=18,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "results_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IOH prediction pipeline")
    parser.add_argument("--csv",        type=str, default=_DEFAULT_CSV,
                        help="Path to model_dataset.csv (or any dataset variant)")
    parser.add_argument("--target",     type=str, default="hypotension_label",
                        help="Target column name")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Where to save results and figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = _dataset_name(args.csv)

    # Confusion-matrix and SHAP filenames are named after the winning model, which
    # can change between runs. Clear any stale ones from a previous run's different
    # winner so the output directory never mixes results from two different models.
    for pattern in ("confusion_matrix_*.png", "shap_summary_*.png"):
        for stale in glob.glob(os.path.join(args.output_dir, pattern)):
            os.remove(stale)

    # ── STEP 1: Load data ─────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1/4 — Loading data")
    print("=" * 70)
    X, y, groups = load_data(csv_path=args.csv, target_column=args.target)
    print(f"Dataset:  {dataset_name}")
    print(f"Loaded {X.shape[0]} episodes, {X.shape[1]} features.")
    print(f"Label distribution: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = split_data(X, y, groups)
    print(f"Train: {X_train.shape[0]} episodes | Test: {X_test.shape[0]} episodes")
    print("(Split is patient-level — all episodes from the same caseid stay together)")

    # ── STEP 2: Train & tune all models ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2/4 — Training & tuning models")
    print("  Models: Logistic Regression, Ridge Regression, Random Forest, XGBoost")
    print("=" * 70)
    results, fitted, bootstrap_aurocs = run_all_models(X_train, X_test, y_train, y_test)

    results_path = os.path.join(args.output_dir, "model_results.csv")
    results.to_csv(results_path, index=False)

    print("\nResults summary:")
    display_cols = [c for c in [
        "model", "status", "roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
        "sensitivity", "specificity", "ppv", "npv", "train_time_s",
    ] if c in results.columns]
    print(results[display_cols].to_string(index=False))

    # ── STEP 3: Combined ROC + AUROC boxplot (all models, every run) ──────────
    print("\n" + "=" * 70)
    print("STEP 3/4 — Saving combined ROC curve and AUROC boxplot")
    print("=" * 70)

    try:
        roc_path = _save_combined_roc(fitted, X_test, y_test, dataset_name, args.output_dir)
        print(f"  Combined ROC   → {roc_path}")
    except Exception as exc:
        print(f"  [warn] Combined ROC failed: {exc}")

    if bootstrap_aurocs:
        try:
            box_path = _save_auroc_boxplot(bootstrap_aurocs, dataset_name, args.output_dir)
            print(f"  AUROC boxplot  → {box_path}")
        except Exception as exc:
            print(f"  [warn] AUROC boxplot failed: {exc}")
    else:
        print("  [warn] No bootstrap AUROCs computed — boxplot skipped.")

    # ── STEP 4: Best model → confusion matrix + results table + SHAP ─────────
    print("\n" + "=" * 70)
    print("STEP 4/4 — Best model: confusion matrix, results table, SHAP")
    print("=" * 70)
    best_name, best_estimator, best_row = select_best_model(
        results, fitted, metric=config.SELECTION_METRIC
    )
    print(f"Best model: {best_name}  "
          f"({config.SELECTION_METRIC} = {best_row[config.SELECTION_METRIC]:.4f})")

    # Confusion matrix — best model only
    try:
        y_pred  = best_estimator.predict(X_test)
        cm_path = _save_confusion_matrix(best_name, y_test, y_pred, args.output_dir)
        print(f"  Confusion matrix → {cm_path}")
    except Exception as exc:
        print(f"  [warn] Confusion matrix failed: {exc}")

    # Results table — all models, every run
    try:
        tbl_path = _save_results_table(results, dataset_name, args.output_dir)
        if tbl_path:
            print(f"  Results table    → {tbl_path}")
    except Exception as exc:
        print(f"  [warn] Results table failed: {exc}")

    # SHAP — best model only
    explanation = explain_best_model(
        best_name, best_estimator, X_train, X_test, y_test, output_dir=args.output_dir
    )
    print(f"  SHAP method: {explanation['method']}")
    print(explanation["summary_df"].head(15).to_string(index=False))
    if explanation.get("plot_path"):
        print(f"  SHAP plot → {explanation['plot_path']}")

    summary_path = os.path.join(args.output_dir, "interpretability_summary.csv")
    explanation["summary_df"].to_csv(summary_path, index=False)

    print("\nDone. All outputs saved in:", args.output_dir)


if __name__ == "__main__":
    main()
