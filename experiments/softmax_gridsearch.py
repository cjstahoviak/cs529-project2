import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pca
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from custom_transformers import WindowSelector
from logistic_regression import SoftmaxRegression

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


def log_pca_plots(pipe, X_train, y_train, params):
    # Fit the PPCA model
    # This is a little hacky, since we are using the pipeline to preprocess the data
    # and then fitting the PPCA model on the preprocessed data
    # the ppca model has plot methods that require the original data
    ppca = pca.pca(params["pca__n_components"], verbose=0)
    win_data = pipe.named_steps["win_selector"].transform(X_train)
    preprocessed_data = pipe.named_steps["scaler"].transform(win_data)
    column_labels = ["_".join(map(str, col)) for col in win_data.columns]
    ppca.fit_transform(preprocessed_data, row_labels=y_train, col_labels=column_labels)

    exp_var_plot, _ = ppca.plot(20)
    biplot, _ = ppca.biplot(labels=y_train, n_feat=10, PC=[0, 1])

    # Save to mlflow
    mlflow.log_figure(biplot, "biplot.png")
    mlflow.log_figure(exp_var_plot, "explained_variance.png")


def log_kaggle_submission(pipe, X_test, timestamp):
    # Predict kaggle data
    y_pred = pipe.predict(X_test)

    # Build required format
    test_results = pd.DataFrame({"class": y_pred}, index=X_test.index)
    test_results.index.name = "id"
    kaggle_submission_fname = f"kaggle_submission_{timestamp}.csv"

    # Save and log
    with tempfile.TemporaryDirectory() as temp_dir:
        kaggle_submission_fname = Path(temp_dir) / kaggle_submission_fname
        test_results.to_csv(kaggle_submission_fname)
        mlflow.log_artifact(kaggle_submission_fname)


def log_confusion_matrix(pipe, X_train, y_train):
    with matplotlib.rc_context(
        {
            "font.size": min(8.0, 50.0 / len(pipe.classes_)),
            "axes.labelsize": 8.0,
            "figure.dpi": 175,
        }
    ):
        disp = ConfusionMatrixDisplay.from_estimator(
            pipe, X_train, y_train, labels=pipe.classes_, normalize="true", cmap="Blues"
        )

        disp.ax_.set_title("Normalized Confusion Matrix")

        mlflow.log_figure(disp.figure_, "confusion_matrix.png")


def main():
    # MLflow setup
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)

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
    lambda_values = [0.01, 0.1, 1, 10]
    tolerance_values = [1e-4, 1e-8, 1e-16]

    base_param_grid = {
        "win_selector__win_size": WIN_SIZES + ["all"],
        "pca__n_components": [20, 0.80, 0.9, 0.99],
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

    param_grids = list(
        ParameterGrid({**base_param_grid, **no_regularization_grid})
    ) + list(ParameterGrid({**base_param_grid, **regularization_grid}))

    # Run grid search
    N_SPLITS = 5
    n_params = len(param_grids)
    print(f"Starting grid search...")
    print(f"n_params: {n_params}, n_splits: {N_SPLITS}")
    print(f"Total number of runs: {n_params * N_SPLITS}")
    for params in tqdm(param_grids, "Running gridsearch"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with mlflow.start_run(run_name=f"softmax_gridsearch_{timestamp}") as run:
            cv = StratifiedShuffleSplit(n_splits=5, random_state=42)
            mlflow.log_params(params)
            pipe.set_params(**params)
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                n_jobs=-2,
                verbose=0,
                return_train_score=True,
            )

            for key, value in scores.items():
                mlflow.log_metric(f"{key}_mean", value.mean())
                mlflow.log_metric(f"{key}_std", value.std())
                for i, val in enumerate(value):
                    mlflow.log_metric(f"{key}_{i}", val)

            # Fit on all data
            start = datetime.now()
            pipe.fit(X_train, y_train)
            end = datetime.now()
            mlflow.log_metric("full_train_time", (end - start).total_seconds())

            # Log the final trained accuracy and loss
            mlflow.log_metric("full_train_accuracy", pipe.score(X_train, y_train))
            full_loss = np.array(pipe.named_steps["logreg"].loss_)
            mlflow.log_metric("full_train_final_loss", full_loss[-1])
            mlflow.log_metric("n_iterations", len(full_loss))
            # mlflow.log_table(pd.DataFrame(full_loss, columns=["loss"]), "loss.json")

            # Log some plots
            log_confusion_matrix(pipe, X_train, y_train)
            # log_pca_plots(pipe, X_train, y_train, params)
            log_kaggle_submission(pipe, X_test, timestamp)
            plt.close("all")


if __name__ == "__main__":
    main()
