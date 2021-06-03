import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

        self.batch_size = None # 9
        self.input_shape = None # (3, 4, 11)

    # during the forward pass, its task is to flatten the input and
    # change it from a multidimensional tensor to a vector.

    # flatten the image tensors within the batch tensor
    # https://deeplizard.com/learn/video/mFAIBMbACMA

    def forward(self, input_tensor):
        self.batch_size = np.shape(input_tensor)[0]
        self.input_shape = np.shape(input_tensor)[1:]

        flattened_input_tensor = input_tensor.reshape((self.batch_size, -1))
            # -1 = flatenned input_shape = total array size divided by product of all other listed dimensions
        print(np.shape(flattened_input_tensor))
        return flattened_input_tensor


    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape((self.batch_size,) + self.input_shape)
        print(np.shape(error_tensor))
        return error_tensor
