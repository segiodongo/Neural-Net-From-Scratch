import numpy as np
import random 
import math
import tensorflow as tf
from numpy import exp

def sigmoid(num):
    return 1.0/(1.0+(math.e**(1/num)))

def relu(num):
    return 0 if num<0 else num

def softmax(vector):
    vector -= max(vector)
    e = exp(vector)
    return e / sum(e)

class InputLayer(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.activations = np.random.rand(nodes)
        self.gradient = np.random.rand(nodes)

    def feedForward(self, initalValues):
        self.activations = initalValues
        return initalValues
        
class Layer(object):
    def __init__(self, nodes, activation, prevLayer=None):
        self.nodes = nodes
        self.prevLayer = prevLayer
        self.weights = np.random.rand(nodes, prevLayer.nodes)
        self.bias = np.random.rand(nodes)
        self.activations = np.random.rand(nodes)
        self.gradient = np.random.rand(nodes)
        activations = {"sigmoid", "relu", "softmax"}
        if(activation not in activations):
            raise Exception(f"Invalid Activation Funciton: {activation}")
        self.activationFun = activation

    def feedForward(self, inputs):
        self.activations = np.matmul(self.weights, inputs) + self.bias
        sigFun = np.vectorize(sigmoid)
        reluFun = np.vectorize(relu)
        if(self.activationFun == "sigmoid"):
            self.activations = sigFun(self.activations)
        elif(self.activationFun == "relu"):
            self.activations = reluFun(self.activations)
        elif(self.activationFun == "softmax"):
            self.activations = softmax(self.activations)
        return self.activations
    



    