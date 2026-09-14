"""Model registry: Logistic Regression, Ridge Regression, Random Forest, XGBoost."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from . import config


@dataclass
class ModelSpec:
    name: str
    build_fn: Callable[[], Any]
    param_distributions: dict = field(default_factory=dict)
    tunable: bool = True
    available: bool = True
    unavailable_reason: Optional[str] = None
    n_iter: Optional[int] = None
    search_n_jobs: int = -1


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def get_model_specs(n_classes: int) -> dict[str, ModelSpec]:
    """Return the registry of four model families."""
    specs: dict[str, ModelSpec] = {}

    # 1. Logistic Regression -------------------------------------------------
    specs["Logistic Regression"] = ModelSpec(
        name="Logistic Regression",
        build_fn=lambda: LogisticRegression(
            max_iter=2000, random_state=config.RANDOM_STATE
        ),
        param_distributions={
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        },
    )

    # 2. Ridge Regression -----------------------------------------------------
    # RidgeClassifier has no predict_proba so we wrap it in CalibratedClassifierCV.
    # alpha is tuned through the wrapper via model__estimator__alpha.
    specs["Ridge Regression"] = ModelSpec(
        name="Ridge Regression",
        build_fn=lambda: CalibratedClassifierCV(
            RidgeClassifier(random_state=config.RANDOM_STATE),
            method="sigmoid",
            cv=config.CV_FOLDS,
        ),
        param_distributions={
            "model__estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        },
    )

    # 3. Random Forest --------------------------------------------------------
    specs["Random Forest"] = ModelSpec(
        name="Random Forest",
        build_fn=lambda: RandomForestClassifier(random_state=config.RANDOM_STATE),
        param_distributions={
            "model__n_estimators": [100, 200, 400, 600],
            "model__max_depth": [None, 4, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    )

    # 4. XGBoost ---------------------------------------------------------------
    XGBClassifier = _try_import_xgboost()
    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        _spw = config.SCALE_POS_WEIGHT
        specs["XGBoost"] = ModelSpec(
            name="XGBoost",
            build_fn=lambda: XGBClassifier(
                objective=objective,
                eval_metric="logloss",
                scale_pos_weight=_spw,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
        )
    else:
        specs["XGBoost"] = ModelSpec(
            name="XGBoost",
            build_fn=lambda: None,
            available=False,
            unavailable_reason="xgboost is not installed (pip install xgboost)",
        )

    return specs
