import numpy as np

from scipy.special import expit as sigmoid

from .optimizers import GD



class LogisticRegressor():
    slots = ['weights', 'bias', 'prob_treshold', 'optimizer']
    
    def __init__(self, p, prob_treshold = 0.5):
        self.p = p
        self.random_init_weights(p)
        self.optimizer = GD(self)
        
        self.prob_treshold = prob_treshold
        
    def random_init_weights(self, p):
        self.weights = np.random.standard_normal(p)
        self.bias = 0
        
    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X):
        return self.predict_proba(X) > self.prob_treshold
    
    def fit(self, X, y, epochs = 10):
        for _ in range(epochs):
            self.optimizer.update(X, y)
        