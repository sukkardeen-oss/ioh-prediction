"""Fits, tunes, and evaluates every model in the registry."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from . import config
from .data_utils import build_preprocessor
from .models import ModelSpec, get_model_specs

# Number of bootstrap resamples used to estimate AUROC variance.
_N_BOOTSTRAP = 1000


def _bootstrap_auroc(y_true, y_proba_pos, n_bootstrap: int = _N_BOOTSTRAP,
                     random_state: int = 42) -> np.ndarray:
    """Resample the test set with replacement and compute AUROC each time.

    Returns an array of length ≤ n_bootstrap (resamples with only one class
    present are discarded to avoid undefined AUROC).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    y_true = np.asarray(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_proba_pos[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, yp))
    return np.array(scores)


def _clinical_metrics(y_true, y_pred) -> dict:
    """Sensitivity, specificity, PPV, and NPV for binary predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv":         tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "npv":         tn / (tn + fn) if (tn + fn) > 0 else np.nan,
    }


def _score_predictions(y_true, y_pred, y_proba, n_classes: int) -> dict:
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
    except Exception:
        metrics["roc_auc"] = np.nan
    if n_classes == 2:
        metrics.update(_clinical_metrics(y_true, y_pred))
    return metrics


def _grid_size(param_distributions: dict) -> int:
    size = 1
    for v in param_distributions.values():
        size *= len(v)
    return size


def _fit_tunable_model(spec: ModelSpec, preprocessor, X_train, y_train, n_classes: int):
    """Build a preprocessing+model pipeline and tune it with RandomizedSearchCV."""
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", spec.build_fn())])

    if not spec.param_distributions:
        pipe.fit(X_train, y_train)
        return pipe, {}

    scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    # Use per-model budget override if set (e.g. LSTM), otherwise global default.
    budget = spec.n_iter if spec.n_iter is not None else config.N_ITER_SEARCH
    n_iter = min(budget, _grid_size(spec.param_distributions))

    search = RandomizedSearchCV(
        pipe,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=config.CV_FOLDS,
        random_state=config.RANDOM_STATE,
        n_jobs=spec.search_n_jobs,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def run_all_models(
    X_train, X_test, y_train, y_test
) -> tuple[pd.DataFrame, dict, dict]:
    """Train + tune every available model.

    Returns
    -------
    results          : pd.DataFrame  — one row per model with all metrics
    fitted           : dict          — model_name → fitted pipeline
    bootstrap_aurocs : dict          — model_name → 1-D array of bootstrap AUROCs
    """
    n_classes = len(np.unique(y_train))
    specs     = get_model_specs(n_classes)

    rows             = []
    fitted           = {}
    bootstrap_aurocs = {}

    for name, spec in specs.items():
        if not spec.available:
            print(f"[skip] {name}: {spec.unavailable_reason}")
            rows.append({"model": name, "status": "unavailable", "note": spec.unavailable_reason})
            continue

        print(f"[train] {name} ...")
        start = time.time()
        try:
            preprocessor          = build_preprocessor(X_train)
            estimator, best_params = _fit_tunable_model(
                spec, preprocessor, X_train, y_train, n_classes
            )
            y_pred  = estimator.predict(X_test)
            y_proba = estimator.predict_proba(X_test)
            metrics = _score_predictions(y_test, y_pred, y_proba, n_classes)

            # Bootstrap AUROC for binary classification only.
            if n_classes == 2:
                boot = _bootstrap_auroc(
                    y_test, y_proba[:, 1],
                    n_bootstrap=_N_BOOTSTRAP,
                    random_state=config.RANDOM_STATE,
                )
                bootstrap_aurocs[name] = boot
                metrics["roc_auc_std"]      = float(boot.std())
                metrics["roc_auc_ci_lower"] = float(np.percentile(boot, 2.5))
                metrics["roc_auc_ci_upper"] = float(np.percentile(boot, 97.5))
        except Exception as exc:
            elapsed = time.time() - start
            print(f"[fail]  {name}: {exc}")
            rows.append({
                "model": name, "status": "failed",
                "note": str(exc)[:300], "train_time_s": round(elapsed, 2),
            })
            continue

        elapsed = time.time() - start
        fitted[name] = estimator
        rows.append({
            "model": name, "status": "ok",
            "train_time_s": round(elapsed, 2),
            "best_params": best_params,
            **metrics,
        })
        ci_str = (
            f"  95% CI [{metrics.get('roc_auc_ci_lower', float('nan')):.4f}–"
            f"{metrics.get('roc_auc_ci_upper', float('nan')):.4f}]"
            if n_classes == 2 else ""
        )
        print(f"[done]  {name}: roc_auc={metrics.get('roc_auc'):.4f}{ci_str}  ({elapsed:.1f}s)")

    return pd.DataFrame(rows), fitted, bootstrap_aurocs


def select_best_model(results: pd.DataFrame, fitted: dict, metric: str = "roc_auc"):
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No model trained successfully; nothing to select.")
    best_row  = ok.sort_values(metric, ascending=False).iloc[0]
    best_name = best_row["model"]
    return best_name, fitted[best_name], best_row
