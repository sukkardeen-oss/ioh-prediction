"""Data loading and preprocessing utilities for the IOH prediction pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Columns present in the IOH CSV that are identifiers, not model features.
_ID_COLS = ["caseid", "episode_number"]

# Columns excluded from the feature matrix
# Organised by the clinical/methodological reason for exclusion.
_LEAKAGE_COLS = [

    # (original) Vasopressors / inotropes
    # Administered specifically to treat hypotension; including them would let
    # the model exploit reverse causality (treatment implies outcome).
    "intraop_phe", "intraop_eph", "intraop_epi", "intraop_ca",

    # (original) Fluids and blood products
    # Given in response to haemodynamic compromise; same reverse-causality risk.
    "intraop_rbc", "intraop_ffp", "intraop_colloid", "intraop_crystalloid",

    # (original) Post-operative outcomes
    # Unavailable at the prediction time-point.
    "icu_days", "death_inhosp",

    # (original) Raw timestamps
    # Encode case ordering and wall-clock time, not physiology.
    "episode_start_sec", "episode_end_sec",
    "casestart", "caseend", "opstart", "opend", "anestart", "aneend",
    "adm", "dis",

    # (original) Subject identifier
    "subjectid",

    # Category 1: near-constant / temporal leakage / unit redundancy
    # nibp_outcome_imputed: True in <0.2 % of episodes — zero discriminative signal.
    "nibp_outcome_imputed",
    # airway: single observed value ("Oral") across the cohort — zero variance.
    "airway",
    # Intraoperative fluid balance measurements recorded as cumulative case-total
    # values. When an arrhythmia episode occurs mid-case the stored value may
    # incorporate output accrued after the prediction horizon (temporal leakage).
    # Urine output additionally risks reverse causality: oliguria is a haemodynamic
    # response to hypotension, not a predictor of it.
    "intraop_ebl",
    "intraop_uo",
    # Intraoperative anaesthetic agents — same temporal leakage rationale as above.
    "intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_rocu", "intraop_vecu",
    # Prothrombin time expressed in redundant units — INR and seconds are
    # collinear with the retained percentage-based measurement (lab_pt%).
    "lab_ptinr", "lab_ptsec",

    # Category 3: clinically irrelevant procedure logistics
    # Peripheral intravenous catheter site: not a haemodynamic predictor.
    "iv1",
    # Endotracheal tube size and its missingness flag: no IOH prediction rationale.
    "tubesize", "tubesize_was_imputed",
    # Cormack–Lehane laryngoscopy grade: airway anatomy, not haemodynamic risk.
    "cormack",

    # Category 4a: missingness indicator flags
    # Encode data-collection patterns rather than physiology; risk introducing
    # systematic bias toward patients with incomplete intraoperative lab panels.
    "intraop_ebl_was_imputed", "intraop_uo_was_imputed",
    "lab_fib_was_imputed", "lab_hco3_was_imputed", "lab_ica_was_imputed",
    "lab_lac_was_imputed", "lab_pco2_was_imputed", "lab_ph_was_imputed",
    "lab_po2_was_imputed", "lab_sao2_was_imputed",

    # Category 4b: intraoperative lab duplicates of pre-operative analytes
    # For analytes measured at both time-points, the intraoperative value is
    # retained only where no pre-operative equivalent exists. Where a preop_*
    # counterpart is available, the lab_* value is dropped to eliminate
    # collinearity and the risk of post-episode blood-draw leakage.
    "lab_hb",    # preop_hb retained
    "lab_plt",   # preop_plt retained
    "lab_na",    # preop_na retained
    "lab_k",     # preop_k retained
    "lab_cr",    # preop_cr retained
    "lab_bun",   # preop_bun retained
    "lab_alb",   # preop_alb retained
    "lab_gluc",  # preop_gluc retained
    "lab_alt",   # preop_alt retained
    "lab_ast",   # preop_ast retained
    "lab_aptt",  # preop_aptt retained
    "lab_pt%",   # preop_pt retained (also redundant with removed lab_ptinr/ptsec)
]

# Minimum number of training-set episodes a categorical level must appear in
# to receive its own one-hot column. Rarer levels are pooled into a single
# "infrequent" category. Applied globally but most impactful for dx (surgical
# diagnosis) and opname, which carry hundreds of low-frequency levels.
_MIN_CATEGORY_FREQUENCY = 20


def load_data(csv_path: str, target_column: str = "hypotension_label"):
    """Load the IOH model dataset from a CSV file.

    Drops identifier columns (caseid, episode_number) from the feature matrix
    but returns caseid separately as the grouping array for patient-level splits.

    Parameters
    ----------
    csv_path:
        Path to model_dataset.csv (or no_rhythm_dataset.csv).
    target_column:
        Name of the binary label column (0 = no IOH, 1 = IOH).

    Returns
    -------
    X      : pd.DataFrame  — feature columns only
    y      : pd.Series     — integer labels (0/1)
    groups : np.ndarray    — caseid per row, for GroupShuffleSplit
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}")

    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    y      = df[target_column].astype(int)
    groups = df["caseid"].to_numpy()

    drop_cols = _ID_COLS + [target_column] + _LEAKAGE_COLS
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y, groups


def split_data(X: pd.DataFrame, y: pd.Series, groups: np.ndarray):
    """Patient-level train/test split via GroupShuffleSplit.

    All episodes from the same patient (caseid) land in the same partition,
    preventing data leakage across the split boundary.
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes + scales numeric columns and
    one-hot encodes categorical columns.

    Rare categorical levels (appearing in fewer than _MIN_CATEGORY_FREQUENCY
    training episodes) are pooled into a single infrequent class rather than
    receiving their own column, preventing the high-cardinality dx and opname
    fields from dominating the encoded feature space with near-empty columns.
    """
    numeric_cols     = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=_MIN_CATEGORY_FREQUENCY,
        )),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer,     numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
