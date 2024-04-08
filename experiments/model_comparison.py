from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from logistic_regression import SoftmaxRegression

train_features_fpath = Path(
    "../data/processed/feature_extracted/train_features_1024.csv"
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

# Defining pipelines for each model
pipelines = {
    "Softmax Regression": make_pipeline(StandardScaler(), PCA(n_components=0.99), SoftmaxRegression(
                learning_rate=0.01,
                learning_rate_decay=0.0,
                max_iter=100_000,
                weight_defaults="zero",
                temperature=1.0,
            ),),
    "Random Forest": make_pipeline(StandardScaler(), PCA(n_components=0.95), RandomForestClassifier(random_state=42)),
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), PCA(n_components=0.95), GaussianNB()),
    "Gradient Boosting Machines": make_pipeline(StandardScaler(), PCA(n_components=0.95), GradientBoostingClassifier(random_state=42)),
    "SVM": make_pipeline(StandardScaler(), PCA(n_components=0.95), SVC(random_state=42))
}

for name, model in pipelines.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} CV accuracy: {np.mean(cv_scores):.3f}")