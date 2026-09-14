"""Shared configuration for the IOH prediction pipeline."""

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Primary metric used for model selection and RandomizedSearchCV scoring.
SELECTION_METRIC = "roc_auc"

# Hyperparameter search budget (used for RandomizedSearchCV where applicable)
N_ITER_SEARCH = 20

# XGBoost class-imbalance correction.
# Derived from model_dataset.csv: n_negative=838, n_positive=446 → ratio ≈ 1.878.
# Passed as scale_pos_weight so the booster up-weights the minority (IOH) class.
SCALE_POS_WEIGHT = 838 / 446
