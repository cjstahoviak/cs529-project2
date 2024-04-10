from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from logistic_regression import SoftmaxRegression

# Load data
train_features_fpath = Path(
    "../data/processed/feature_extracted/train_features_2048.csv"
).resolve()
train_features_df = pd.read_csv(train_features_fpath, index_col=0)
X_train = train_features_df.drop(["target"], axis=1)
X_train.columns = X_train.columns + "_" + X_train.iloc[0] + "_" + X_train.iloc[1]
X_train = X_train.iloc[2:]
X_train = X_train.apply(pd.to_numeric, errors="coerce")

y_train = train_features_df["target"].iloc[2:]
X_test = X_train
y_test = y_train

print(X_train.head())
print("Total features in X_train: " + str(X_train.shape[1]))
print("Total instances in X_train: " + str(X_train.shape[0]))
# print("First element: " + str(X_train.iloc[0, 0]))

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
softmax_pipe.fit(X_train, y_train)
y_pred = softmax_pipe.predict(X_test)
accuracy = softmax_pipe.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
