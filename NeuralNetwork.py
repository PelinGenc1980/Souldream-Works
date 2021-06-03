import copy
import numpy as np
from Layers.Base import BaseLayer




class NeuralNetwork(BaseLayer):

    def __init__(self, optimizer, weights_initializer, bias_initializer):


        self.optimizer = optimizer
        self.loss : list = []  # loss value for each iteration after calling train
        self.layers : list = [] # architecture
        self.data_layer = None  # input data and labels
        self.loss_layer = None  # special layer > loss and prediction
        self.input = None
        self.label = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):

        self.input, self.label = self.data_layer.forward()
        input_tensor = self.input

        for layer in self.layers:
            print('layer', layer)
            # output tensor of the former layer = input tensor of the next layer
            input_tensor = layer.forward(input_tensor)

        loss = self.loss_layer.forward(input_tensor, self.label)
        print('loss:', loss)

        return loss

    def backward(self):

        # starting from the loss layer passing it the label tensor for the current input

        error = self.loss_layer.backward(self.label)

        # backpropagation for every layer from back to front
        for layer in self.layers[::-1]:
            error = layer.backward(error)
            #print('error', error)

        return error

    def append_layer(self, layer):

        if layer.trainable:  ##########################################################################
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)

        print('self.layers', np.shape(self.layers))
        self.layers.append(layer)


    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()
        print('iterations:', iterations)   # iterations = 4000 ????????? input tensor = (150,4)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        logits = input_tensor

        print('logits:', logits)
        print('np.shape(logits):', np.shape(logits))

        return logits









