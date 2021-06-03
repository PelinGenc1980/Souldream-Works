import numpy as np

class Sgd:    #  Stochastic Gradient Descent

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - np.dot(self.learning_rate, gradient_tensor)
        return updated_weights

class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.update_v = 0  # which value should it take at first??????????????????

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.update_v = self.momentum_rate * self.update_v - self.learning_rate * gradient_tensor

        updated_weights = weight_tensor + self.update_v

        return updated_weights


class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu  # beta 1
        self.rho = rho # beta 2

        self.update_v = 0
        self.update_r = 0

        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1

        self.update_v = self.mu * self.update_v  + (1 - self.mu) * gradient_tensor
        self.update_r = self.rho * self.update_r + (1 - self.rho) * gradient_tensor ** 2

        # bias correction
        self.hat_v = self.update_v / (1 - self.mu ** self.k)
        self.hat_r = self.update_r / (1 - self.rho ** self.k)

        updated_weights = weight_tensor - self.learning_rate * self.hat_v / (np.sqrt(self.hat_r) + np.finfo(float).eps )

        return updated_weights


