import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    # Cost function for softmax regression
    def _compute_cost(self, X, y, coef_):
        m = X.shape[0]
        z = np.dot(X, coef_)
        y_hat = self._softmax(z)
        cost = (-1 / m) * np.sum(y * np.log(y_hat))
        return cost

    def _gradient_descent(self, X, y, coef_, learning_rate, iterations):
        m = X.shape[0]
        cost_history = []

        for i in range(iterations):
            z = np.dot(X, coef_)
            y_hat = self._sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (y_hat - y))

            coef_ -= learning_rate * dw

            cost = self._compute_cost(X, y, coef_)
            cost_history.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")

        return coef_, cost_history

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # Initialize weights
        self.coef_ = np.zeros(X.shape[1])

        # Gradient descent
        # for _ in range(self.max_iter):
        # linear_model = np.dot(X, self.coef_) + self.intercept_
        # y_pred = self._sigmoid(linear_model)
        # error = y_pred - y

        # # Update weights
        # self.coef_ -= self.learning_rate * np.dot(X.T, error) / X.shape[0]
        # self.intercept_ -= self.learning_rate * np.sum(error) / X.shape[0]

        self._compute_cost(X, y, self.coef_)
        self.coef_, cost_history = self._gradient_descent(
            X, y, self.coef_, self.learning_rate, self.max_iter
        )

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        linear_model = np.dot(X, self.coef_)
        return self._sigmoid(linear_model)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
