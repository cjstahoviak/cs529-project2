import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def _softmax(self, Z):
        # Subtract max for numerical stability
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.array(X)  # Ensure `X` is a NumPy array

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize weights and bias
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes_):
            y_one_hot[:, i] = y == c

        # Gradient descent
        for _ in range(self.max_iter):
            # scores = np.dot(X, self.weights) + self.bias
            scores = X.dot(self.weights) + self.bias
            probabilities = self._softmax(scores)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_one_hot))
            db = (1 / n_samples) * np.sum(probabilities - y_one_hot, axis=0)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        scores = np.dot(X, self.weights) + self.bias
        probabilities = self._softmax(scores)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        predictions = self.predict(X)
        return accuracy_score(y, predictions)
