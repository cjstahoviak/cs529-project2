import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from custom_transformers import WindowSelector
from logistic_regression import SoftmaxRegression
import os

N_JOBS = os.cpu_count() * 2
EXPERIMENT_NAME = "/cs529_project_2_softmax_gridsearch"
DATA_FOLDER_PATH = Path("../data/processed/feature_extracted/pickle").resolve()
WIN_SIZES = [1024, 2048, 4096, 8192]


def load_data():
    train_dict = {}
    test_dict = {}

    y_train = None

    for win_size in WIN_SIZES:
        train_df: pd.DataFrame = pd.read_pickle(
            DATA_FOLDER_PATH / f"train_features_{win_size}.pkl"
        )
        X_test: pd.DataFrame = pd.read_pickle(
            DATA_FOLDER_PATH / f"test_features_{win_size}.pkl"
        )

        X_train = train_df.drop(columns=["target"], level=0)

        train_dict[win_size] = X_train
        test_dict[win_size] = X_test

        if y_train is None:
            y_train = train_df["target"].values.ravel()

    X_train = pd.concat(
        train_dict.values(),
        axis=1,
        keys=WIN_SIZES,
        names=["win_size", "feature", "stat"],
    )
    X_test = pd.concat(
        test_dict.values(),
        axis=1,
        keys=WIN_SIZES,
        names=["win_size", "feature", "stat"],
    )

    return X_train, y_train, X_test


def main():
    # MLflow setup
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog(log_datasets=False, log_models=False)

    # Load data
    print("Loading data...")
    X_train, y_train, X_test = load_data()

    print("Data loaded:")
    print(f"X_train:\n{X_train.head()}")
    print(f"y_train:\n{y_train}")
    print(f"X_test:\n{X_test.head()}")

    # Define pipeline
    print("Creating pipeline...")
    pipe = Pipeline(
        [
            ("win_selector", WindowSelector()),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            (
                "logreg",
                SoftmaxRegression(
                    verbose=0, max_iter=100_000, regularization="l2", lam=0.1
                ),
            ),
        ]
    )

    # Create parameter grids to search over
    lambda_values = [0.01, 0.1, 0.05, 1]
    tolerance_values = [1e-4, 1e-8]

    base_param_grid = {
        "win_selector__win_size": WIN_SIZES,
        "pca__n_components": [0.80, 0.99],
        "logreg__tol": tolerance_values,
    }

    # Create two separate grids for regularization and no regularization
    regularization_grid = {
        "logreg__regularization": ["l2", "l1"],
        "logreg__lam": lambda_values,
    }
    no_regularization_grid = {
        "logreg__regularization": [None],
    }

    param_grids = [
        {**base_param_grid, **no_regularization_grid},
        {**base_param_grid, **regularization_grid},
    ]

    # Run grid search
    N_SPLITS = 5

    gc = GridSearchCV(
        pipe,
        param_grid=param_grids,
        n_jobs=N_JOBS,
        cv=StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=42),
        verbose=2,
        pre_dispatch="n_jobs",  # Reduce memory consumption
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=f"softmax_gridsearch_{timestamp}"):

        # Fit the model
        gc.fit(X_train, y_train)

        # Run best model on kaggle test data
        y_pred = gc.best_estimator_.predict(X_test)
        test_results = pd.DataFrame({"class": y_pred}, index=X_test.index)
        test_results.index.name = "id"

        # Save kaggle test results
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_submission_fname = Path(tmpdir) / f"kaggle_submission_{timestamp}.csv"
            test_results.to_csv(kaggle_submission_fname)
            mlflow.log_artifact(kaggle_submission_fname)


if __name__ == "__main__":
    main()
