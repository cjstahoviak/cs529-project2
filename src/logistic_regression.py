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

    def _softmax(self, logits):
        # TODO: Implement temperature hyperparameter

        # Subtract the maximum value from the logits to prevent overflow
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _init_weights_and_bias(self, n_features, n_classes):
        if self.weight_defaults == "zero":
            return np.zeros((n_features, n_classes)), np.zeros(n_classes)
        elif self.weight_defaults == "random":
            return np.random.randn(n_features, n_classes), np.random.randn(n_classes)
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

        # Validate input X and y are correctly sized
        X, y = check_X_y(X, y)

        # Encode target labels
        self.y_encoder_ = LabelBinarizer()
        y_one_hot = self.y_encoder_.fit_transform(y)

        self.classes_ = self.y_encoder_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize weights and bias
        self.weights_, self.bias_ = self._init_weights_and_bias(n_features, n_classes)

        # Run optimization
        self._optimize(X, y_one_hot)

        return self

    def _compute_probabilities(self, X):
        logits = np.dot(X, self.weights_) + self.bias_
        probabilities = self._softmax(logits)
        return probabilities

    def _optimize(self, X, y_one_hot):
        n_instances = X.shape[0]

        self.loss_ = []
        current_iter = 0

        while current_iter < self.max_iter:
            probabilities = self._compute_probabilities(X)

            # Compute gradients of loss function with respect to logits, weights and bias
            grad_logits = probabilities - y_one_hot
            grad_weights = (1 / n_instances) * np.dot(X.T, grad_logits)
            grad_bias = (1 / n_instances) * np.sum(grad_logits, axis=0)

            # Update weights and bias
            self.weights_ -= self.learning_rate * grad_weights
            self.bias_ -= self.learning_rate * grad_bias

            # TODO: Implement learning rate decay

            # Track loss and loss-change
            current_loss = -np.mean(
                np.sum(y_one_hot * np.log(probabilities + 1e-9), axis=1)
            )

            self.loss_.append(current_loss)

            # Calculate and print change in loss if not the first iteration
            if current_iter > 0:
                loss_change = self.loss_[current_iter - 1] - self.loss_[current_iter]
                if current_iter % 100 and self.verbose:
                    print(
                        f"Iteration {current_iter:6}: Loss {self.loss_[current_iter]:10.6f}, Change in Loss {loss_change:10.6f}"
                    )
                if loss_change < self.tol:
                    if self.verbose:
                        print(f"Converged after {current_iter} iterations.")
                    return
            else:
                if current_iter % 100 and self.verbose:
                    print(
                        f"Iteration {current_iter:6}: Loss {self.loss_[current_iter]:10.6f}"
                    )

            current_iter += 1

        if self.verbose:
            print("Maximum number of iterations reached.")

    def predict_proba(self, X):
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
