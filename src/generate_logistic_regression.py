from pathlib import Path

import pandas as pd

from logistic_regression import CustomLogisticRegression

# Load data
# train_fpath = Path("../data/processed/train_data.pkl").resolve()
# test_fpath = Path("../data/processed/test_data.pkl").resolve()
# train_df = pd.read_pickle(train_fpath)
# test_df = pd.read_pickle(test_fpath)
# y_train = train_df["target"]
# X_train = train_df["audio"]
# X_test = test_df["audio"]

train_features_fpath = Path(
    "../data/processed/feature_extracted/train_features_512.csv"
).resolve()
train_features_df = pd.read_csv(train_features_fpath, index_col=0)
X_train = train_features_df.drop(["target"], axis=1)
X_train.columns = X_train.columns + "_" + X_train.iloc[0] + "_" + X_train.iloc[1]
X_train = X_train.iloc[2:]
X_train = X_train.apply(pd.to_numeric, errors="coerce")

y_train = train_features_df["target"].iloc[2:]
X_test = X_train

print(X_train.head())
# print(X_train.info())
# print(y_train.info())
# X_test = X_train
# print(X_train.dtypes)

# Fit model and predict
clr = CustomLogisticRegression(learning_rate=0.01, max_iter=10_000)
clr.fit(X_train, y_train)
clr.predict(X_train)

# Print accuracy
accuracy = clr.score(X_train, y_train)
print(f"Our Accuracy: {accuracy}")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=10_000)
lr.fit(X_train, y_train)
lr.predict(X_train)

# Print accuracy
accuracy = lr.score(X_train, y_train)
print(f"Sklearn Accuracy: {accuracy}")
