"""Interpretability via SHAP for all four model families.

- Logistic Regression  -> LinearExplainer (exact, fast)
- Ridge Regression     -> LinearExplainer on averaged fold coefficients
                          (reaches through CalibratedClassifierCV wrapper)
- Random Forest        -> TreeExplainer (exact, fast)
- XGBoost              -> TreeExplainer with tree_path_dependent perturbation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHAP_MODELS = {
    "Logistic Regression",
    "Ridge Regression",
    "Random Forest",
    "XGBoost",
}

_EXACT_BACKGROUND, _EXACT_SAMPLE = 100, 200


def _get_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transform.__self__.n_features_in_)]


def explain_best_model(best_name, estimator, X_train, X_test, y_test, output_dir="."):
    """Run SHAP for the winning model.

    Returns {"method": ..., "summary_df": pd.DataFrame, "plot_path": str | None}.
    """
    if best_name in SHAP_MODELS:
        return _explain_with_shap(best_name, estimator, X_train, X_test, output_dir)
    raise ValueError(f"No interpretability method registered for '{best_name}'")


def _explain_with_shap(best_name, estimator, X_train, X_test, output_dir):
    import shap

    preprocessor  = estimator.named_steps["preprocessor"]
    model         = estimator.named_steps["model"]

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    feature_names = _get_feature_names(preprocessor)
    # Strip ColumnTransformer prefixes added by sklearn (num__, cat__)
    feature_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    background  = shap.sample(X_train_t, min(_EXACT_BACKGROUND, len(X_train_t)), random_state=0)
    sample_test = X_test_t[: min(_EXACT_SAMPLE, len(X_test_t))]

    if best_name == "Random Forest":
        explainer   = shap.TreeExplainer(model, data=background, feature_names=feature_names)
        shap_values = explainer(sample_test)
        values      = shap_values.values

    elif best_name == "XGBoost":
        explainer   = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent", feature_names=feature_names
        )
        shap_values = explainer(sample_test)
        values      = shap_values.values

    elif best_name == "Logistic Regression":
        explainer   = shap.LinearExplainer(model, background, feature_names=feature_names)
        shap_values = explainer(sample_test)
        values      = shap_values.values

    elif best_name == "Ridge Regression":
        values = _ridge_shap_values(model, background, sample_test, feature_names)

    else:
        raise ValueError(f"No SHAP path defined for '{best_name}'")

    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs   = np.abs(values).mean(axis=0)
    summary_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plot_path = f"{output_dir}/shap_summary_{best_name.replace(' ', '_')}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(values, sample_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as exc:
        plot_path = None
        print(f"[warn] could not save SHAP plot: {exc}")

    return {"method": "shap", "summary_df": summary_df, "plot_path": plot_path}


def _ridge_shap_values(calibrated_model, background, sample_test, feature_names):
    """Exact SHAP values for Ridge Regression via averaged fold coefficients."""
    import shap

    base_models = [cc.estimator for cc in calibrated_model.calibrated_classifiers_]
    coef      = np.atleast_2d(np.mean([m.coef_ for m in base_models], axis=0))
    intercept = np.mean([np.atleast_1d(m.intercept_) for m in base_models], axis=0)

    per_class = []
    for k in range(coef.shape[0]):
        explainer = shap.LinearExplainer(
            (coef[k], float(intercept[k])), background, feature_names=feature_names
        )
        per_class.append(explainer(sample_test).values)

    if len(per_class) == 1:
        return per_class[0]
    return np.stack(per_class, axis=-1)
