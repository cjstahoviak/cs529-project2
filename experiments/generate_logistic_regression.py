from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logistic_regression import SoftmaxRegression


def main():
    parser = ArgumentParser(prog="Run Logistic Regression on Extracted Audio Features")
    parser.add_argument(
        "--data_fpath",
        "-d",
        type=Path,
        help="Path to the training features pickle file (.pkl)",
        required=True,
    )

    args = parser.parse_args()
    data_fpath = args.data_fpath.resolve()
    df = pd.read_pickle(data_fpath)
    X = df.drop(["target"], axis=1)
    y = df["target"]

    print("Loaded data")
    print(X.head())

    # Define pipeline and score
    softmax_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            (
                "softmaxreg",
                SoftmaxRegression(
                    learning_rate=0.01,
                    learning_rate_decay=0.0,
                    max_iter=10_000,
                    weight_defaults="zero",
                    temperature=1.0,
                ),
            ),
        ]
    )

    print("Initialized pipeline")
    print(softmax_pipe)
    print("Fitting pipeline")
    start = datetime.now()
    softmax_pipe.fit(X, y)
    end = datetime.now()
    print(f"Fit pipeline in {end - start}")
    accuracy = softmax_pipe.score(X, y)
    print(f"Train Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
