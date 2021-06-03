import numpy as np
from Layers.Base import BaseLayer





class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.output_size = output_size

        self._optimizer = None
        self.grad_weights = None


        self.weights = np.random.uniform(0, 1, (self.input_size+1, self.output_size))
        self.input = None


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @property
    def gradient_weights(self):
        return self.grad_weights

    def forward(self, input_tensor):
        # with bias
        print('input_tensor:',np.shape(input_tensor))
        bias = np.ones((input_tensor.shape[0], 1))
        self.input = np.hstack((input_tensor, bias))

        logits = self.input @ self.weights
        #print('output_tensor:', np.shape(output_tensor))
        return logits    #output tensor = error tensor

    def backward(self, error_tensor):

        weights_b = self.weights[0:-1, :] #weights_b v: weight without bias

        error = error_tensor @ weights_b.T   # E @ W.T

        #print('error:', np.shape(error))
####################################################################################################

       # Gradient tensor to calculate updated weights
        self.grad_weights = self.input.T @ error_tensor   # X.T @ E

        # update weights
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            #print('weights:', np.shape(self.weights))

        return error

    def initialize(self, weights_initializer, bias_initializer):

        self.shape = (self.output_size, self.input_size)

        self.w = weights_initializer.initialize(self.shape, self.shape[1], self.shape[0])
        self.b = bias_initializer.initialize((1,self.input_size),self.shape[1], self.shape[0])

        self.weights = np.hstack((self.w, self.b))
