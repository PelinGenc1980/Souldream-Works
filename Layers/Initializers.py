import numpy as np

class Constant:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        init_weight = self.constant * np.ones(weights_shape)
        print("weights_shape", weights_shape)

        return init_weight


class UniformRandom:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        init_weight = np.random.uniform(0, 1, weights_shape)

        return init_weight

# sigma = sqrt(2/(fan_out+fan_in))
# Why does this initialization help prevent gradient problems?
# It sets the weight matrix neither too bigger than 1, nor too smaller than 1.
# Thus it doesn’t explode or vanish gradients respectively.
class Xavier:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_out + fan_in))
        init_weight = np.random.normal(loc=0, scale=sigma, size=weights_shape)

        return init_weight

# sigma = sqrt(2/fan_in)
class He:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        init_weight = np.random.normal(loc=0, scale=sigma, size=weights_shape)

        return init_weight

