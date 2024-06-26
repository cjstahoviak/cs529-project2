import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression model for multi-class classification. Inheriting from scikit-learn's BaseEstimator and ClassifierMixin classes.

    Parameters:
    - learning_rate: float, the learning rate for gradient descent (default=0.01)
    - learning_rate_decay: float, the learning rate decay factor (default=0.0)
    - max_iter: int, the maximum number of iterations for optimization (default=1000)
    - weight_defaults: str, the weight initialization method ("zero" or "random") (default="zero")
    - regularization: str, the type of regularization ("l1", "l2", or None) (default=None)
    - lam: float, the regularization parameter (default=1)
    - temperature: float, the temperature scaling factor (default=1.0)
    - verbose: int, the verbosity level (0 for no output, 1 for loss and loss-change) (default=0)
    - tol: float, the tolerance for convergence (default=1e-4)

    Attributes:
    - learning_rate: float, the learning rate for gradient descent
    - learning_rate_decay: float, the learning rate decay factor
    - max_iter: int, the maximum number of iterations for optimization
    - weight_defaults: str, the weight initialization method
    - temperature: float, the temperature scaling factor
    - verbose: int, the verbosity level
    - tol: float, the tolerance for convergence
    - regularization: str, the type of regularization
    - lam: float, the regularization parameter
    - initial_learning_rate: float, the initial learning rate
    - y_encoder_: LabelBinarizer, the label binarizer for encoding target labels
    - classes_: numpy array, the unique classes in the target labels
    - weights_: numpy array, the optimized weights
    - bias_: numpy array, the optimized bias
    - loss_: list, the loss values during optimization

    Methods:
    - fit(X, y): Fit the Softmax Regression model to the training data
    - predict_proba(X): Compute the class probabilities for the input features
    - predict(X): Predict the class labels for the input features
    - score(X, y): Compute the accuracy score of the model on the given test data
    """

    def __init__(
        self,
        learning_rate=0.01,
        learning_rate_decay=0.0,  # No decay by default
        max_iter=1000,
        weight_defaults="zero",
        regularization=None,
        lam=1,
        temperature=1.0,  # No temperature scaling by default
        verbose=0,
        tol=1e-4,
    ):
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_iter = max_iter
        self.weight_defaults = weight_defaults
        self.temperature = temperature
        self.verbose = verbose  # Prints loss and loss-change
        self.tol = tol
        self.regularization = regularization
        self.lam = lam
        self.initial_learning_rate = learning_rate

    # Rest of the code...
class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        learning_rate_decay=0.0,  # No decay by default
        max_iter=1000,
        weight_defaults="zero",
        regularization=None,
        lam=1,
        temperature=1.0,  # No temperature scaling by default
        verbose=0,
        tol=1e-4,
    ):
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_iter = max_iter
        self.weight_defaults = weight_defaults
        self.temperature = temperature
        self.verbose = verbose  # Prints loss and loss-change
        self.tol = tol
        self.regularization = regularization
        self.lam = lam
        self.initial_learning_rate = learning_rate

    def _softmax(self, logits):
        """
        Compute the softmax function for the given logits.

        Parameters:
        - logits: numpy array of shape (n_samples, n_classes)

        Returns:
        - probabilities: numpy array of shape (n_samples, n_classes)
        """
        # Normalization: Subtract the maximum value from the logits to prevent overflow/underflow
        exp_scores = (
            np.exp((logits - np.max(logits, axis=1, keepdims=True))) / self.temperature
        )
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _init_weights(self, n_features, n_classes):
        """
        Initialize the weights for the softmax regression model.

        Parameters:
        - n_features: int, the number of input features
        - n_classes: int, the number of output classes

        Returns:
        - weights: numpy array of shape (n_features + 1, n_classes), the initialized weights
        """
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
        """
        Fit the Softmax Regression model to the training data.

        Parameters:
        - X: numpy array or pandas DataFrame, the input features
        - y: numpy array or pandas Series, the target labels

        Returns:
        - self: fitted Softmax Regression model
        """

        if self.verbose:
            print("Fitting Softmax Regression model...")
            print(self.get_params())

        # Convert X and y to numpy arrays if they are pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Encode target labels
        self.y_encoder_ = LabelBinarizer()
        y_one_hot = self.y_encoder_.fit_transform(y)

        self.classes_ = self.y_encoder_.classes_

        # Optimize weights using gradient descent
        self.weights_, self.bias_, self.loss_ = self._gd_optimize(X, y_one_hot)

        return self

    def _compute_probabilities(self, X):
        """
        Compute the probabilities for each class given the input features.

        Parameters:
        - X: numpy array or pandas DataFrame, the input features

        Returns:
        - probabilities: numpy array of shape (n_samples, n_classes), the class probabilities
        """
        logits = np.dot(X, self.weights_) + self.bias_
        probabilities = self._softmax(logits)
        return probabilities

    def _compute_loss(self, probabilities, y_one_hot):
        """
        Compute the loss function for the Softmax Regression model.

        Parameters:
        - probabilities: numpy array of shape (n_samples, n_classes), the class probabilities
        - y_one_hot: numpy array of shape (n_samples, n_classes), the one-hot encoded target labels

        Returns:
        - loss: float, the computed loss
        """
        return -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-9), axis=1))

    def _regularization_loss_term(self, weight):
        """
        Compute the regularization loss term for the Softmax Regression model.

        Parameters:
        - weight: numpy array of shape (n_features + 1, n_classes), the weights

        Returns:
        - loss_term: float, the computed regularization loss term
        """
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
        """
        Compute the regularization gradient term for the Softmax Regression model.

        Parameters:
        - weight: numpy array of shape (n_features + 1, n_classes), the weights

        Returns:
        - gradient_term: numpy array of shape (n_features + 1, n_classes), the computed regularization gradient term
        """
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
        """
        Optimize weights using gradient descent.

        Parameters:
        - X: numpy array or pandas DataFrame, the input features
        - y_one_hot: numpy array of shape (n_samples, n_classes), the one-hot encoded target labels

        Returns:
        - weights: numpy array of shape (n_features, n_classes), the optimized weights
        - bias: numpy array of shape (1, n_classes), the optimized bias
        - loss: list, the loss values during optimization
        """
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

            # Calculate the updated learning rate
            self.learning_rate = self.initial_learning_rate / (
                1 + self.learning_rate_decay * current_iter
            )

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
        """
        Compute the class probabilities for the input features.

        Parameters:
        - X: numpy array or pandas DataFrame, the input features

        Returns:
        - probabilities: numpy array of shape (n_samples, n_classes), the class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._compute_probabilities(X)

    def predict(self, X):
        """
        Predict the class labels for the input features.

        Parameters:
        - X: numpy array or pandas DataFrame, the input features

        Returns:
        - predictions: numpy array, the predicted class labels
        """
        probabilities = self.predict_proba(X)
        return self.y_encoder_.inverse_transform(probabilities)

    def score(self, X, y):
        """
        Compute the accuracy score of the model on the given test data.

        Parameters:
        - X: numpy array or pandas DataFrame, the input features
        - y: numpy array or pandas Series, the target labels

        Returns:
        - score: float, the accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
