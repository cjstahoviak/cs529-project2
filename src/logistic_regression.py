import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        weight_defaults="zero",
        regularization=None,
        lam=1,
        temperature=1.0,
        verbose=0,
        tol=1e-4,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weight_defaults = weight_defaults
        self.temperature = temperature
        self.verbose = verbose  # Prints loss and loss-change
        self.tol = tol
        self.regularization = regularization
        self.lam = lam

    def _softmax(self, logits):
        # Normalization: Subtract the maximum value from the logits to prevent overflow/underflow
        exp_scores = (
            np.exp((logits - np.max(logits, axis=1, keepdims=True))) / self.temperature
        )
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _init_weights(self, n_features, n_classes):
        shape = (n_features + 1, n_classes)
        if self.weight_defaults == "zero":
            return np.zeros(shape)
        elif self.weight_defaults == "random":
            return np.random.randn(shape)
        else:
            raise ValueError(
                f"Invalid weight initialization: {self.weight_defaults}. Must be 'zero' or 'random'."
            )

    def fit(self, X, y):

        if self.verbose:
            print("Fitting Softmax Regression model...")
            print(self.get_params())

        # Convert X and y to numpy arrays if they are pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Validate input X and y are correctly sized with bias term
        # maybe _x_internal variable ?
        X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        X_with_bias, y = check_X_y(X, y)

        # Encode target labels
        self.y_encoder_ = LabelBinarizer()
        y_one_hot = self.y_encoder_.fit_transform(y)

        self.classes_ = self.y_encoder_.classes_

        # Optimize weights using gradient descent
        self.weights_, self.bias_, self.loss_ = self._gd_optimize(X, y_one_hot)

        return self

    def _compute_probabilities(self, X):
        logits = np.dot(X, self.weights_) + self.bias_
        probabilities = self._softmax(logits)
        return probabilities

    def _compute_loss(self, probabilities, y_one_hot):
        return -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-9), axis=1))

    def _regularization_loss_term(self, weight):
        if self.lam == 0 or self.regularization is None:
            return 0
        elif self.regularization == "l1":
            return self.lam * np.sum(np.abs(weight[:-1]))
        elif self.regularization == "l2":
            return self.lam * np.sum(weight[:-1] ** 2)
        else:
            raise ValueError(
                f"Invalid regularization: {self.regularization}. Must be 'l1' or 'l2' or none."
            )

    def _regularization_gradient_term(self, weight):
        n_classes = weight.shape[1]
        if self.lam == 0 or self.regularization is None:
            return 0
        elif self.regularization == "l1":
            return np.vstack(
                [self.lam * np.sign(weight[:-1]), np.zeros((1, n_classes))]
            )
        elif self.regularization == "l2":
            return 2 * self.lam * np.vstack([weight[:-1], np.zeros((1, n_classes))])
        else:
            raise ValueError(
                f"Invalid regularization: {self.regularization}. Must be 'l1' or 'l2' or none."
            )

    def _gd_optimize(self, X, y_one_hot):
        n_instances, n_features = X.shape
        n_classes = y_one_hot.shape[1]

        X = np.c_[X, np.ones(n_instances)]  # Add bias term
        weight = self._init_weights(n_features, n_classes)

        loss_change = np.inf
        self.loss_ = []
        current_iter = 0

        while current_iter < self.max_iter and self.tol < loss_change:
            logits = np.dot(X, weight)
            probabilities = self._softmax(logits)

            self.loss_.append(
                self._compute_loss(probabilities, y_one_hot)
                + self._regularization_loss_term(weight)
            )

            gradient = (1 / n_instances) * np.dot(
                X.T, probabilities - y_one_hot
            ) + self._regularization_gradient_term(weight)

            weight -= self.learning_rate * gradient

            # TODO: Implement learning rate decay

            if current_iter > 0:
                loss_change = self.loss_[current_iter - 1] - self.loss_[current_iter]

            if (not (current_iter % 100)) and self.verbose:
                print(
                    f"Iteration {current_iter:6}: Loss {self.loss_[current_iter]:10.6f}, Change in Loss {loss_change:10.6f}"
                )

            current_iter += 1

        if self.verbose:
            if current_iter == self.max_iter:
                print(
                    f"Optimization stopped after reaching the maximum number of iterations. Final loss: {self.loss_[-1]}"
                )
            else:
                print(
                    f"Optimization converged in {current_iter} iterations. Final loss: {self.loss_[-1]}"
                )

        return weight[:-1, :], weight[-1, :], self.loss_

    def predict_proba(self, X):
        # X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        check_is_fitted(self)

        X = check_array(X)

        return self._compute_probabilities(X)

    def predict(self, X):

        # Select most probable class
        probabilities = self.predict_proba(X)
        return self.y_encoder_.inverse_transform(probabilities)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
