import numpy as np
from layer import *



class Model(object):

    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)
    
    def feedForward(self, inputValues):
        values = inputValues
        i=0
        for layer in self.layers:
            i+=1
            # if(i==2): print(f"Layer {i} mean: ", layer.activations.mean())
            values = layer.feedForward(values)
        return values

    def calculateLoss(self, output, target):
        difference = np.subtract(output, target)
        square = np.square(difference)
        mse = square.mean()
        return mse

    def evaluate(self, inputs, target):
        result = self.feedForward(inputs)
        loss = self.calculateLoss(result, target)
        if(np.argmax(result) == np.argmax(target)):
            return (loss, 1)
        else:
            return (loss, 0)
    
    def predict(self, input):
        output = self.feedForward(input)
        return np.argmax(output)


