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

n = 1000

a = 3

def generate_lr_data(n, beta):
    y = np.random.binomial(1, 0.5, n)

    X = np.random.normal(0, 1, (n, 2))
    X_mod = np.column_stack([np.ones(n), X]) # intercept
    X_mod = np.column_stack([X_mod, X[:, 0] * X[:, 1]]) # interactions

    Z = np.dot(X_mod, beta)
    y = np.random.binomial(1, sigmoid(Z))

    return X, y

np.random.seed(44)
X, y = generate_lr_data(n, [1, 5, 5, 1])

data = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "target"])
data.to_csv("data/artificial.csv")
np.mean(y)

plt.scatter(X[y==0, 0], X[y == 0, 1], s=20, edgecolor='k')
plt.scatter(X[y==1, 0], X[y == 1, 1], s=20, edgecolor='k')
plt.suptitle("Artificial data for classification", fontsize=16)
plt.legend(["y = 0", "y = 1"])
plt.savefig("plots/artificial_data/classes.png")

plt.show()


results = compare_methods(X, y, ["LDA", "QDA", "decision tree", "random forest"], k=5, test_size = 0.2)

plot_boundaries(X, y, "LDA", save_plot = True)
plot_boundaries(X, y, "QDA", save_plot = True)
plot_boundaries(X, y, "decision tree", save_plot = True)
plot_boundaries(X, y, "random forest", save_plot = True)
plot_boundaries(X, y, "logistic regression", "adam", save_plot = True)
plot_boundaries(X, y, "logistic regression", "iwls", save_plot = True)
plot_boundaries(X, y, "logistic regression", "sgd", save_plot = True)

plot_boundaries(X, y, "logistic regression", "adam", True, save_plot = True)
plot_boundaries(X, y, "logistic regression", "iwls", True, save_plot = True)
plot_boundaries(X, y, "logistic regression", "sgd", True, save_plot = True)




