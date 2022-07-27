import numpy as np
import random 
import math
import tensorflow as tf
from numpy import exp

#Function declarations for activation functions
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
        #Number of nodes in the layer
        self.nodes = nodes
        
        #Initializes Random Values for activations and gradient vector
        self.activations = np.random.rand(nodes)
        self.gradient = np.random.rand(nodes)
    #Function for the network to feed forward its values to the next layer
    def feedForward(self, initalValues):
        self.activations = initalValues
        return initalValues
        
class Layer(object):
    def __init__(self, nodes, activation, prevLayer=None):

        #Number of nodes in the layer
        self.nodes = nodes
        
        #A reference to the previous layer
        self.prevLayer = prevLayer

        #initializes random arrays for weights biases, activations and gradient vector
        self.weights = np.random.rand(nodes, prevLayer.nodes)
        self.bias = np.random.rand(nodes)
        self.activations = np.random.rand(nodes)
        self.gradient = np.random.rand(nodes)

        #Initializes activation function for layer
        activations = {"sigmoid", "relu", "softmax"}
        if(activation not in activations):
            raise Exception(f"Invalid Activation Funciton: {activation}")
        self.activationFun = activation

    def feedForward(self, inputs):
        #Calculates values to be passed into activation function
        self.activations = np.matmul(self.weights, inputs) + self.bias

        #Alows function to be ran on each element of a vector
        sigFun = np.vectorize(sigmoid)
        reluFun = np.vectorize(relu)

        #Applies the correct activation function to the activations of layer
        if(self.activationFun == "sigmoid"):
            self.activations = sigFun(self.activations)
        elif(self.activationFun == "relu"):
            self.activations = reluFun(self.activations)
        elif(self.activationFun == "softmax"):
            self.activations = softmax(self.activations)
        return self.activations
    



    