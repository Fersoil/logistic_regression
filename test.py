import numpy as np
import pandas as pd

import logistic_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

aids = pd.read_csv('data/aids.csv')

y = aids['target']
X = aids.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

lr = logistic_regression.LogisticRegressor()
import pdb; pdb.set_trace()
X_copy_train = lr.create_data_frame(X_train)

X_train.shape[1]