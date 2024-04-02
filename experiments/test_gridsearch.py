from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlxtend.feature_selection import ColumnSelector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from custom_transformers import ElementwiseSummaryStats, LibrosaTransformer

# MLflow setup
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/cs529_project_2")
mlflow.autolog(log_datasets=False, log_models=False)

# Load data
train_fpath = Path("../data/processed/train_data.pkl").resolve()
test_fpath = Path("../data/processed/test_data.pkl").resolve()

train_df = pd.read_pickle(train_fpath)
test_df = pd.read_pickle(test_fpath)

y_train = train_df["target"]
X_train = train_df["audio"]

X_test = test_df["audio"]

# Define pipeline
pipe = Pipeline(
    [
        ("librosa_features", LibrosaTransformer()),
        ("summary_stats", ElementwiseSummaryStats(desc_kw_args={"axis": 1})),
        ("stat_selector", ColumnSelector()),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        (
            "logreg",
            LogisticRegression(
                penalty=None, max_iter=10_000, solver="saga", multi_class="multinomial"
            ),
        ),
    ]
)

# Define parameter grid
WIN_SIZES = [1024, 2048, 4096]

# Parameters for librosa feature extraction
# These must be separated to prevent parameters that don't make sense together
librosa_params_grid = [
    {"feature": ["mfcc"], "n_mfcc": [13, 20, 30, 40], "n_fft": WIN_SIZES},
    {"feature": ["chroma_stft"], "n_fft": WIN_SIZES},
    {"feature": ["zero_crossing_rate"], "frame_length": WIN_SIZES},
    {"feature": ["spectral_centroid"], "n_fft": WIN_SIZES},
    {"feature": ["spectral_bandwidth"], "n_fft": WIN_SIZES},
    {"feature": ["spectral_rolloff"], "n_fft": WIN_SIZES},
    {"feature": ["spectral_contrast"], "n_fft": WIN_SIZES},
]

# Parameters that are the same for all grid search iterations
base_param_grid = {
    "librosa_features__n_fft": WIN_SIZES,
    "stat_selector__cols": [["mean", "variance"], ["mean", "variance", "min", "max"]],
    "pca__n_components": [0.85, 0.95],
}

# Combine base parameters with librosa parameters
param_grid = []
for params in librosa_params_grid:
    params = {f"librosa_features__{k}": v for k, v in params.items()}
    param_grid.append({**base_param_grid, **params})


gc = GridSearchCV(
    pipe,
    param_grid=param_grid,
    n_jobs=-1,
    cv=StratifiedShuffleSplit(n_splits=3, random_state=42),
    verbose=2,
    pre_dispatch="n_jobs",  # Reduce memory consumption
)

with mlflow.start_run(run_name="sklearn_logreg_gridsearch"):
    gc.fit(X_train, y_train)

    y_pred = gc.best_estimator_.predict(X_test)

    test_results = pd.DataFrame({"class": y_pred}, index=test_df.index)
    test_results.index.name = "id"
    test_results.to_csv("test_results.csv")
    mlflow.log_artifact("test_results.csv")
