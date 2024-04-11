import os
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.custom_transformers import WindowSelector
from src.logistic_regression import SoftmaxRegression

N_JOBS = 1
EXPERIMENT_NAME = "/cs529_project_2_softmax_gridsearch"
DATA_FOLDER_PATH = Path("../data/processed/feature_extracted/pickle").resolve()
WIN_SIZES = [1024, 2048, 4096, 8192]


def load_data():
    """
    Load the training and testing data from pickle files.

    Returns:
        X_train (pd.DataFrame): Training data features.
        y_train (np.ndarray): Training data target.
        X_test (pd.DataFrame): Testing data features.
    """
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
            ("classifier", None),
        ]
    )

    # Create parameter grids to search over
    base_param_grid = {
        "win_selector__win_size": WIN_SIZES + ["All"],
        "pca__n_components": [0.80, 0.9, 0.99],
    }

    classifier_grids = [
        {
            "classifier": [SoftmaxRegression()],
            "classifier__regularization": ["l2"],
            "classifier__lam": [0.01, 0.1, 0.05, 1],
            "classifier__tol": [1e-4, 1e-8],
        },
        {
            "classifier": [RandomForestClassifier()],
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, 15],
        },
        {
            "classifier": [GaussianNB()],
        },
        {
            "classifier": [GradientBoostingClassifier()],
            "classifier__n_estimators": [100, 200, 300],
            "classifier__loss": ["log_loss", "deviance", "exponential"],
            "classifier__learning_rate": [0.01, 0.1, 0.5],
        },
        {
            "classifier": [SVC()],
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "poly", "rbf"],
        },
    ]

    param_grids = [
        {**base_param_grid, **classifier_grid} for classifier_grid in classifier_grids
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
            kaggle_submission_fname = (
                Path(tmpdir) / f"kaggle_submission_{timestamp}.csv"
            )
            test_results.to_csv(kaggle_submission_fname)
            mlflow.log_artifact(kaggle_submission_fname)


if __name__ == "__main__":
    main()
