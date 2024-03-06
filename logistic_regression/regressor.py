import numpy as np

from scipy.special import expit as sigmoid

from .optimizers import mini_batch_gd, iwls, adam, sgd, newton


class LogisticRegressor:
    slots = ["beta", "prob_threshold", "optimize"]

    def __init__(self, p, descent_algorithm="minibatch", prob_threshold=0.5):
        """
        Initialize the LogisticRegressor class.

        Parameters:
        - p (int): The number of features in the input data.
        - descent_algorithm (str, optional): The descent algorithm to use for optimization. Defaults to "minibatch". Options are "minibatch", "newton", "iwls", "adam", "sgd".
        - prob_threshold (float, optional): The probability threshold for classification, which determines w Defaults to 0.5.
        """

        self.p = p
        self.random_init_weights(p)
        self.descent_algorithm = descent_algorithm

        self.prob_threshold = prob_threshold

    def random_init_weights(self, p):
        self.beta = np.random.standard_normal(p)

    def predict_proba(self, X):
        return sigmoid(X @ self.beta)

    def predict(self, X):
        return self.predict_proba(X) > self.prob_threshold

    def minus_log_likelihood(self, X, y):
        weighted_input = X @ self.beta
        L = np.sum(y * weighted_input - np.log(1 + np.exp(weighted_input)))
        return -L

    def loss(self, y, y_hat_proba):
        # log likelihood loss
        return -np.sum(y * np.log(y_hat_proba) + (1 - y) * np.log(1 - y_hat_proba))

    def loss_prime(X, y, beta):
        """
        calculates the derivative of the loss function with respect to the beta
        """
        # as we know from MSO
        p = sigmoid(X @ beta)
        if y.shape == ():  # if y is a scalar, then there wont be matrix multiplication
            return -X.T * (y - p)
        return -X.T @ (y - p)

    def loss_second(X, y, beta):
        """
        calculates the second derivative of the loss function with respect to the beta
        """
        # as we know from MSO

        p = sigmoid(X @ beta)
        W = np.diag(p * (1 - p))
        return X.T @ W @ X

    def fit(
        self, X, y, learning_rate=0.01, max_num_epoch=1000, batch_size=32, verbose=False
    ):
        # TODO normalize the data

        # TODO interactions

        if self.descent_algorithm == "minibatch":
            self.beta = mini_batch_gd(
                X,
                y,
                self.beta,
                LogisticRegressor.loss_prime,
                learning_rate=learning_rate,
                max_num_epoch=max_num_epoch,
                batch_size=batch_size,
                verbose=verbose,
            )

        elif self.descent_algorithm == "iwls":
            self.beta = iwls(
                X,
                y,
                self.beta,
                max_num_epoch=max_num_epoch,
                verbose=verbose,
            )
        elif self.descent_algorithm == "adam":
            self.beta = adam(
                X,
                y,
                self.beta,
                LogisticRegressor.loss_prime,
                learning_rate=learning_rate,
                max_num_epoch=max_num_epoch,
                verbose=verbose,
            )
        elif self.descent_algorithm == "sgd":
            self.beta = sgd(
                X,
                y,
                self.beta,
                LogisticRegressor.loss_prime,
                learning_rate=learning_rate,
                max_num_epoch=max_num_epoch,
                verbose=verbose,
            )
        elif self.descent_algorithm == "newton":
            self.beta = newton(
                X,
                y,
                self.beta,
                LogisticRegressor.loss_prime,
                LogisticRegressor.loss_second,
                max_num_epoch=max_num_epoch,
                verbose=verbose,
            )
        else:
            raise ValueError("Invalid descent_algorithm")

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def balanced_accuracy(self, X, y):
        y_hat = self.predict(X)
        tp = np.sum(y_hat & y)
        tn = np.sum(~y_hat & ~y)
        fp = np.sum(y_hat & ~y)
        fn = np.sum(~y_hat & y)
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))
