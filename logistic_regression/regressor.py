import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from scipy.special import expit as sigmoid

from .optimizers import mini_batch_gd, iwls, adam, sgd, newton


class LogisticRegressor:
    slots = ["beta", "prob_threshold", "descent_algorithm", "include_interactions"]

    def __init__(
        self,
        descent_algorithm="minibatch",
        prob_threshold=0.5,
        include_interactions=False,
    ):
        """
        Initialize the LogisticRegressor class.

        Parameters:
        - descent_algorithm (str, optional): The descent algorithm to use for optimization. Defaults to "minibatch". Options are "minibatch", "newton", "iwls", "adam", "sgd".
        - prob_threshold (float, optional): The probability threshold for classification, which determines w Defaults to 0.5.
        - include_interactions (bool, optional): Whether to include interaction terms in the model. Defaults to False.
        """

        self.descent_algorithm = descent_algorithm
        self.prob_threshold = prob_threshold
        self.include_interactions = include_interactions
        self.beta = None

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
        self,
        X,
        y,
        learning_rate=0.01,
        max_num_epoch=1000,
        batch_size=32,
        verbose=False,
    ):
        # TODO normalize the data

        if self.include_interactions:
            # initialize the weights
            self.random_init_weights(X.shape[1] + X.shape[1] * (X.shape[1] - 1) // 2)
            X = self.create_data_frame_with_interactions(X)
        else:
            self.random_init_weights(X.shape[1])

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

    def create_data_frame_with_interactions(self, X):
        # create the new column names
        #! add "intercept" at the beginning of the list of column names when modifying
        col_names_X = list(X.columns)
        new_col_names = col_names_X.copy()
        for idx, first_variable_name in enumerate(col_names_X):
            for second_variable_name in col_names_X[idx + 1 :]:
                new_col_names.append(f"{first_variable_name}*{second_variable_name}")

        # create the interaction terms
        #! for now without the intercept but its fairly easy to add,
        #! removing include_bias=False created columns of 1s at the beginning of the dataframe
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X = poly.fit_transform(X)
        X = pd.DataFrame(X, columns=new_col_names)
        return X
