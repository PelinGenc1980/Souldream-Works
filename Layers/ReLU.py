import numpy as np
from Layers.Base import BaseLayer
from Optimization import Optimizers

class ReLU(BaseLayer):   #Rectified Linear Unit


    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input = None


    def forward(self, input_tensor):
        self.input = input_tensor

        #if input_tensor > 0, then input tensor
        #else 0
        output = np.where(input_tensor > 0, input_tensor, np.zeros_like(input_tensor))

        return output


    def backward(self, error_tensor):

        error = np.where(self.input <= 0, np.zeros_like(error_tensor), error_tensor)  # (12)

        return error

    # RELU gradient has a constant value results in faster learning.
    # In contrast, the gradient of sigmoids becomes increasingly small
    # as the absolute value of x increases. Vanishing gradient problem.
