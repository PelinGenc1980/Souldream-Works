import numpy as np


class CrossEntropyLoss:


    def __init__(self):
        self.y_pred = None


    def forward(self, input_tensor, label_tensor):
        self.y_pred = input_tensor


        eps = np.finfo(input_tensor.dtype).eps   # np.finfo: easy way to get epsilon
        input_tensor = np.maximum(input_tensor, eps)

        loss = np.sum(-np.log(input_tensor) * label_tensor)  # (15)

        return loss

    # Derivative of loss function
    def backward(self, label_tensor):

        error = -np.divide(label_tensor, self.y_pred)   # (16)
        return error
