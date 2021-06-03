import numpy as np
from Layers.Base import BaseLayer
#from Optimization import Optimizers


class SoftMax(BaseLayer):


    def __init__(self):
        super().__init__()
        self.trainable = False




    def forward(self, input_tensor):

        # shift xk for stability : x_tilda = xk - max(xk). This ensures that the largest input element is 0, keeping us safe from overflows. Even better, it doesn't affect the gradient calculation at all.
        f = input_tensor - np.max(input_tensor)
        sum_f = np.sum(np.exp(f), axis=1, keepdims=True)  # keepdims = True returned a 2D array
        self.S = np.exp(f) / sum_f

        return self.S

    # probabilities of logits add up to 1

    # E(n-1) = yhat * (En - sum(En,j * yj_hat))
    def backward(self, error_tensor):
        A = error_tensor
        B = self.S

        #print('predicted y:', np.shape(B))
        #print('error_tensor:', np.shape(A))
        #print('(A * B).sum(axis=1)[:, None]:', np.shape((A * B).sum(axis=1)[:, None]))

        error = B * (A - (A * B).sum(axis=1)[:, None])
        #https://sgugger.github.io/a-simple-neural-net-in-numpy.html

        return error





