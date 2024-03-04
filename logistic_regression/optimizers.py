from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from scipy.special import expit as sigmoid


def mini_batch_gd(
    X, y, initial_solution, calculate_gradient, learning_rate=0.01, max_num_epoch=1000, batch_size=1, batch_fraction=None, verbose=False
):
    """
    Performs mini batch gradient descent optimization.

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_weights: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - learning_rate: Learning rate for updating the solution (default: 0.01).
    - max_num_iters: Maximum number of iterations (default: 1000).
    - batch_size: Size of the mini batch (default: 1).
    - batch_fraction: Fraction of the data to use in each mini batch (default: None).
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy().T
    current_solution = initial_solution

    # set batch size
    assert type(batch_size) is int, "batch_size must be an integer"
    if batch_fraction is not None:
        assert 0 < batch_fraction <= 1, "batch_fraction must be between 0 and 1"
        batch_size = int(X.shape[0] * batch_fraction)
    iterations = int(X.shape[0] / batch_size)

    for epoch in range(max_num_epoch):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        X, y = X[shuffled_idx], y[shuffled_idx]
        for idx in range(iterations):
            X_selected, y_selected = X[idx * batch_size : (idx + 1) * batch_size], y[idx * batch_size : (idx + 1) * batch_size]
            gradient = calculate_gradient(X_selected, y_selected, current_solution)
            current_solution = current_solution - learning_rate * gradient
        print(f"Epoch {epoch}, solution:", current_solution)
    return current_solution

def newton(X, y, initial_solution, calculate_gradient, calculate_hessian, max_num_epoch=1000, verbose=False):
    """
    Performs Newton method optimization using second order derviatives

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_weights: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - calculate_hessian: Function to calculate the Hessian.
    - max_num_epoch: Maximum number of iterations (default: 1000).
    - verbose: Whether to print the solution at each iteration (default: False).


    Returns:
    - The optimized solution.
    """

    # initialization
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy().T
    current_solution = initial_solution

    for epoch in range(max_num_epoch):
        gradient = calculate_gradient(X, y, current_solution)
        hessian = calculate_hessian(X, y, current_solution)
        current_solution = current_solution - np.linalg.inv(hessian) @ gradient
        if verbose:
            print(f"Epoch {epoch}, solution:", current_solution)
    return current_solution

def iwls(X, y, initial_solution, max_num_epoch=1000, verbose=False):
    """
    Performs iteratively reweighted least squares optimization. Uses the log-likelihood loss

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_solution: Initial solution for optimization.
    - max_num_epoch: Maximum number of iterations (default: 1000).
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy().T
    current_solution = initial_solution

    for epoch in range(max_num_epoch):
        P = sigmoid(X @ current_solution)
        W = np.diag(P * (1 - P))
        Z = X @ current_solution + np.linalg.inv(W) @ (y - P)
        current_solution = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Z
        if verbose:
            print(f"Epoch {epoch}, solution:", current_solution)
    return current_solution



def sgd(X, y, initial_solution, calculate_gradient, learning_rate=0.01, max_num_epoch=1000, verbose=False):
    """
    Performs stochastic gradient descent optimization.

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_solution: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - learning_rate: Learning rate for updating the solution (default: 0.01).
    - max_num_iters: Maximum number of iterations (default: 1000).
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy().T
    current_solution = initial_solution 

    for epoch in range(max_num_epoch):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        X, y = X[shuffled_idx], y[shuffled_idx]
        for X_selected, y_selected in zip(X, y):
            gradient = calculate_gradient(X_selected, y_selected, current_solution)
            current_solution = current_solution - learning_rate * gradient
        if verbose:
            print(f"Epoch {epoch}, solution: {current_solution}")
    return current_solution



def adam(
    X,
    y,
    initial_solution,
    calculate_gradient,
    learning_rate=0.01,
    momentum_decay=0.9,
    squared_gradient_decay=0.99,
    max_num_epoch=1000,
    batch_size=1,
    batch_fraction=None,
    epsilon=1e-8,
    verbose=False,
):
    """
    Performs optimization with adam algorithm.

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_solution: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - learning_rate: Learning rate for updating the solution (default: 0.01).
    - momentum_decay: Decay rate for the momentum (default: 0.9).
    - squared_gradient_decay: Decay rate for the squared gradient (default: 0.99).
    - max_num_iters: Maximum number of iterations (default: 1000).
    - batch_size: Size of the mini batch (default: 1).
    - batch_fraction: Fraction of the data to use in each mini batch (default: None).
    - epsilon: Small value to avoid division by zero (default: 1e-8).
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy().T
    current_solution = initial_solution
    momentum = np.zeros_like(initial_solution)
    squared_gradients = np.zeros_like(initial_solution)
    counter = 0

    # set batch size
    assert type(batch_size) is int, "batch_size must be an integer"
    if batch_fraction is not None:
        assert 0 < batch_fraction <= 1, "batch_fraction must be between 0 and 1"
        batch_size = int(X.shape[0] * batch_fraction)
    iterations = int(X.shape[0] / batch_size)

    for epoch in range(max_num_epoch):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        X, y = X[shuffled_idx], y[shuffled_idx]
        for idx in range(iterations):
            X_selected, y_selected = (
                X[idx * batch_size : (idx + 1) * batch_size],
                y[idx * batch_size : (idx + 1) * batch_size],
            )
            gradient = calculate_gradient(X_selected, y_selected, current_solution)
            momentum = momentum_decay * momentum + (1 - momentum_decay) * gradient
            squared_gradients = (
                squared_gradient_decay * squared_gradients
                + (1 - squared_gradient_decay) * gradient**2
            )
            counter += 1

            # bias correction
            corrected_momentum = momentum / (1 - momentum_decay**counter)
            corrected_squared_gradients = squared_gradients / (1 - squared_gradient_decay**counter)

            current_solution = current_solution - learning_rate * corrected_momentum / (
                np.sqrt(corrected_squared_gradients) + epsilon
            )

        print(f"Epoch {epoch}, solution:", current_solution)
    return current_solution
