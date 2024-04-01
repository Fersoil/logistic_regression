from logistic_regression import compare_methods, plot_boundaries
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.special import expit as sigmoid


aids = pd.read_csv('data/aids.csv')
X = aids.drop('target', axis=1)
y = aids['target']

results = compare_methods(X, y, ["LDA", "QDA", "decision tree", "random forest"], k=5, test_size = 0.2)

pd.DataFrame(results).plot(kind='box')
plt.title("Balanced accuracy of different classification models")
plt.show()


# generate random data

n = 10000

a = 3

def generate_qda_data(n, p, a, ro):
    y = np.random.binomial(1, 0.5, n)

    cov_matrix = np.array([[1, ro], [ro, 1]])
    cov_matrix2 = np.array([[1, -ro], [-ro, 1]])
    X = np.multiply(np.random.multivariate_normal([0, 0], cov_matrix, n), 1 - y[:, np.newaxis]) + np.multiply(np.random.multivariate_normal([a, a], cov_matrix2, n), y[:, np.newaxis])
    return X, y

X, y = generate_qda_data(n, 2, a, 0.5)
np.mean(y)

results = compare_methods(X, y, ["LDA", "QDA", "decision tree", "random forest"], k=5, test_size = 0.2)

plot_boundaries(X, y, "LDA", save_plot = True)
plot_boundaries(X, y, "QDA", save_plot = True)
plot_boundaries(X, y, "decision tree", save_plot = True)
plot_boundaries(X, y, "random forest", save_plot = True)
plot_boundaries(X, y, "logistic regression", "adam", save_plot = True)
plot_boundaries(X, y, "logistic regression", "iwls", save_plot = True)
plot_boundaries(X, y, "logistic regression", "sgd", save_plot = True)




