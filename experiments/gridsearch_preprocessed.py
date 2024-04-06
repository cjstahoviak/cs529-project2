import os
from datetime import datetime
from pathlib import Path

import mlflow
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
CURRENT_EXPERIMENT = mlflow.set_experiment("/cs529_project_2")
mlflow.autolog(log_datasets=False, log_models=False, extra_tags={"current": True})

WIN_SIZE = 2048

# Load data
print("Loading data...")
train_fpath = Path(
    f"../data/processed/feature_extracted/train_features_{WIN_SIZE}.csv"
).resolve()
test_fpath = Path(
    f"../data/processed/feature_extracted/test_features_{WIN_SIZE}.csv"
).resolve()

train_df = pd.read_csv(train_fpath, header=[0, 1, 2], index_col=0)
test_df = pd.read_csv(test_fpath, header=[0, 1, 2], index_col=0)

X_train = train_df.drop(columns=["target"], level=0)
y_train = train_df["target"].values.ravel()

X_test = test_df

# Define pipeline
print("Creating pipeline...")
pipe = Pipeline(
    [
        # ("stat_selector", ColumnSelector(cols = (slice(None), slice(None), ))),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        (
            "logreg",
            LogisticRegression(
                penalty="l1", max_iter=10_000, solver="saga", multi_class="multinomial"
            ),
        ),
    ]
)

param_grid = {
    "pca__n_components": [0.75, 0.85, 0.9, 0.95, 0.99],
}

gc = GridSearchCV(
    pipe,
    param_grid=param_grid,
    n_jobs=-1,
    cv=StratifiedShuffleSplit(n_splits=5, random_state=42),
    verbose=2,
    pre_dispatch="n_jobs",  # Reduce memory consumption
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


def getChildRuns(run_id):
    experiment_ids = [CURRENT_EXPERIMENT.experiment_id]
    for potential_child_run in mlflow.search_runs(
        experiment_ids=experiment_ids, output_format="list"
    ):
        parent = mlflow.get_parent_run(potential_child_run.info.run_id)
        if parent and parent.info.run_id == run_id:
            yield potential_child_run


parent_run_id = None
with mlflow.start_run(
    run_name=f"sklearn_logreg_gridsearch_preprocessed_{timestamp}"
) as parent_run:
    parent_run_id = parent_run.info.run_id
    mlflow.log_param("win_size", WIN_SIZE)
    gc.fit(X_train, y_train)

    # Run best model on kaggle test data
    y_pred = gc.best_estimator_.predict(X_test)
    test_results = pd.DataFrame({"class": y_pred}, index=test_df.index)
    test_results.index.name = "id"

    kaggle_submission_fname = f"kaggle_submission_{timestamp}.csv"
    test_results.to_csv(kaggle_submission_fname)
    mlflow.log_artifact(kaggle_submission_fname)
    os.remove(kaggle_submission_fname)


for child_run in getChildRuns(parent_run_id):
    with mlflow.start_run(run_id=child_run.info.run_id):
        mlflow.log_param("win_size", WIN_SIZE)
