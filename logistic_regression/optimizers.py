from abc import ABC, abstractmethod
import numpy as np

from scipy.special import expit as sigmoid


class Optimizer(ABC):
    def __init__(self, Regressor):
        self.Regressor = Regressor
        
    def loss(self, y_hat, y):
        # mse - for temporary testing purposes
        #return np.mean((y - y_hat) ** 2)
        # cross-entropy
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)

        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def loss_prime(self, y_hat, y):
        #return np.mean(2 * (y - y_hat))
        # cross-entropy
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
        return np.mean(- y / y_hat + (1 - y) / (1 - y_hat))
    
    def sigmoid_prime(self, x):
        return sigmoid(x) * (1 - sigmoid(x))

    @abstractmethod
    def update(self, layer):
        pass

    
class GD(Optimizer):
    def __init__(self, Regressor, learning_rate = 0.01):
        self.learning_rate = learning_rate
        super().__init__(Regressor)
    
    def update(self, X, y):
        
        batch_size = X.shape[1]
        
        weighted_input = X @ self.Regressor.weights + self.Regressor.bias
        y_hat_proba = sigmoid(weighted_input)
        
        g = self.loss_prime(y_hat_proba, y)
        print("loss prime", np.sum(g)/ batch_size)
        g = g * self.sigmoid_prime(weighted_input)
        print("sigmoid prime", np.sum(g)/ batch_size)
        
        db = np.sum(g) / batch_size
        
        dw = g.dot(X) / batch_size
        
        self.Regressor.weights -= self.learning_rate * dw
        self.Regressor.bias -= -self.learning_rate * db
        
        return self.loss(y_hat_proba, y)
