import numpy as np
import math
import tensorflow as tf

def sigmoid(num):
    return 1.0/(1.0+(math.e**(1/num)))


def calculateMetrics(model, images, labels):
    assert(len(images) == len(labels))
    totalCost = 0
    accuracy = 0
    for i, image in enumerate(images):
        label = labels[i]
        cost, accurate = model.evaluate(image, label)
        totalCost += cost
        accuracy += accurate
    return (totalCost / len(images) * 100, accuracy / len(images)*100)

def splitIntoBatches(data, labels, batchSize):
    numBatches = len(data)//batchSize
    batches = []
    for i in range(numBatches):
        batches.append((data[i*batchSize:(i+1)*batchSize], labels[i*batchSize:(i+1)*batchSize]))
    lastData = data[(numBatches)*batchSize:]
    lastLabels = labels[(numBatches)*batchSize:]
    batches.append((lastData, lastLabels))
    return batches

def calculateActivationDerivatives(activation):
    for i, a in enumerate(activation):
        activation[i] = 0 if a<0 else 1
    return activation

def reverseSig(a):
    return a * (1- a)

def calculateSigmoidDerivatives(activation):
    for i, a in enumerate(activation):
        activation[i] = reverseSig(a)
    return activation

def updateWeights(model, learningRate):
    for i,layer in enumerate(model.layers):
        if(i!=0):
            prevLayer = model.layers[i-1]
            gradient = layer.gradient.reshape((len(layer.activations), 1))
            inputs = prevLayer.activations.reshape((1, len(prevLayer.activations)))
            layer.weights -= np.matmul(gradient, inputs) * learningRate
            layer.bias -= learningRate * layer.gradient
        
    
def backPropogate(model, target):
    for i in range(len(model.layers)-1, 0, -1):
        layer=model.layers[i]
        if(i==len(model.layers)-1):
            errors = layer.activations - target
        else:
            higherLayer = model.layers[i+1]
            errors = np.matmul(higherLayer.gradient, higherLayer.weights)
        derivates = calculateActivationDerivatives(np.copy(layer.activations))
        layer.gradient = np.multiply(errors, derivates)

def trainModel(model, data, labels, epochs = 3):
    batchSize = 128
    learningRate = 0.0001
    batches = splitIntoBatches(data, labels, batchSize)
    # print(type(batches[0]))
    for epoch in range(epochs):
        for i, batch in enumerate(batches):
            for j,input in enumerate(batch[0]):
                if(len(batch[0])>0):
                    label = batch[1][j]
                    model.feedForward(input)
                    backPropogate(model, label)
                    updateWeights(model, learningRate)
                    # print(model.layers[1].gradient[0])
            if(len(batch[0])>0):
                performance, accuracy = calculateMetrics(model, batch[0], batch[1])
            
            print(f"Batch # {i+1},   accuracy: {accuracy},   Loss: {performance}")
        print("###################################")
        print("###################################")
        print("###################################")
        print("Epoch :", epoch+1)
        print("###################################")
        print("###################################")
        print("###################################")

    


