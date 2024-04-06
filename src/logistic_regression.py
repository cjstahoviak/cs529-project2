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
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weight_defaults = "zero"
        self.temperature = 1.0
        self.verbose = 0  # Prints loss and loss-change

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
        # List hyper-paramters
        print("Training with:")
        print(f"\tLearning rate: {self.learning_rate}")
        print(f"\tMax iterations: {self.max_iter}")
        print(f"\tWeight initialization: {self.weight_defaults}")
        print(f"\tTemperature: {self.temperature}")

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
        n_instances, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize weights and bias
        self.weights_, self.bias_ = self._init_weights_and_bias(n_features, n_classes)

        prev_loss = None  # For tracking loss over time
        for i in range(self.max_iter):
            logits = np.dot(X, self.weights_) + self.bias_
            probabilities = self._softmax(logits)

            # Compute gradients of loss function with respect to logits, weights and bias
            grad_logits = probabilities - y_one_hot
            grad_weights = (1 / n_instances) * np.dot(X.T, grad_logits)
            grad_bias = (1 / n_instances) * np.sum(grad_logits, axis=0)

            # Update weights and bias
            self.weights_ -= self.learning_rate * grad_weights
            self.bias_ -= self.learning_rate * grad_bias

            # TODO: Implement learning rate decay

            # Track loss and loss-change
            loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-9), axis=1))
            if (
                prev_loss is not None
            ):  # Calculate and print change in loss if not the first iteration
                loss_change = loss - prev_loss
                if i % 100 and self.verbose:
                    print(
                        f"Iteration {i:6}: Loss {loss:10.6f}, Change in Loss {loss_change:10.6f}"
                    )
            else:
                if i % 100 and self.verbose:
                    print(f"Iteration {i:6}: Loss {loss:10.6f}")
            prev_loss = loss  # Update the previous loss with the current loss

        print("Training complete.")
        return self

    def predict_proba(self, X):
        check_is_fitted(self)

        X = check_array(X)
        scores = np.dot(X, self.weights_) + self.bias_
        probabilities = self._softmax(scores)

        return probabilities

    def predict(self, X):
        # Select most probable class
        probabilities = self.predict_proba(X)
        return self.y_encoder_.inverse_transform(probabilities)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
