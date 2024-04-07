from pathlib import Path

import pandas as pd

from logistic_regression import SoftmaxRegression

# train_features_fpath = Path(
#     "../data/processed/feature_extracted/train_features_8192.csv"
# ).resolve()
# train_features_df = pd.read_csv(train_features_fpath, index_col=0)
# X_train = train_features_df.drop(["target"], axis=1)
# X_train.columns = X_train.columns + "_" + X_train.iloc[0] + "_" + X_train.iloc[1]
# X_train = X_train.iloc[2:]
# X_train = X_train.apply(pd.to_numeric, errors="coerce")

# y_train = train_features_df["target"].iloc[2:]
# X_test = X_train
# y_test = y_train

# Load data
WIN_SIZE = 2048
print("Loading data...")
data_dir = Path("../data/processed/feature_extracted/pickle").resolve()
train_df: pd.DataFrame = pd.read_pickle(data_dir / f"train_features_{WIN_SIZE}.pkl")
X_test: pd.DataFrame = pd.read_pickle(data_dir / f"test_features_{WIN_SIZE}.pkl")
X_train = train_df.drop(columns=["target"], level=0)
y_train = train_df["target"].values.ravel()

X_test = X_train
y_test = y_train

print(X_train.head())
print("Total features in X_train: " + str(X_train.shape[1]))
print("Total instances in X_train: " + str(X_train.shape[0]))
# print("First element: " + str(X_train.iloc[0, 0]))

# Fit model
sr = SoftmaxRegression(
    learning_rate=0.0001,
    max_iter=1_000,
    weight_defaults="zero",
    temperature=1.0,
    verbose=0,
)
sr.fit(X_train, y_train)

# Print accuracy
accuracy = sr.score(X_test, y_test)
print(f"Our Accuracy: {accuracy}")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    penalty=None, max_iter=1_000, solver="saga", multi_class="multinomial", verbose=0
)
lr.fit(X_train, y_train)

# Print accuracy
accuracy = lr.score(X_test, y_test)
print(f"Sklearn Accuracy: {accuracy}")
