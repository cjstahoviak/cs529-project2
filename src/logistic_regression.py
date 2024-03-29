import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    # def _sigmoid(self, z):
    #     return 1 / (1 + np.exp(-z))

    # def fit(self, X, y):
    #     X, y = check_X_y(X, y)

    #     # Initialize weights
    #     self.coef_ = np.zeros(X.shape[1])
    #     self.intercept_ = 0

    #     # Gradient descent
    #     for _ in range(self.max_iter):
    #         linear_model = np.dot(X, self.coef_) + self.intercept_
    #         y_pred = self._sigmoid(linear_model)
    #         error = y_pred - y

    #         # Update weights
    #         self.coef_ -= self.learning_rate * np.dot(X.T, error) / X.shape[0]
    #         self.intercept_ -= self.learning_rate * np.sum(error) / X.shape[0]

    #     return self

    # def predict_proba(self, X):
    #     check_is_fitted(self)
    #     X = check_array(X)

    #     linear_model = np.dot(X, self.coef_) + self.intercept_
    #     return self._sigmoid(linear_model)

    # def predict(self, X):
    #     probabilities = self.predict_proba(X)
    #     return (probabilities >= 0.5).astype(int)
