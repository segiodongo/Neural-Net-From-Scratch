import numpy as np
import math
import tensorflow as tf

#Reference sigmoid function
def sigmoid(num):
    return 1.0/(1.0+(math.e**(1/num)))

#Calculates the accuracy and average loss of a model on a batch of data
def calculateMetrics(model, images, labels):
    assert(len(images) == len(labels))
    totalCost = 0
    accuracy = 0
    for i, image in enumerate(images):
        label = labels[i]
        cost, accurate = model.evaluate(image, label)
        totalCost += cost
        accuracy += accurate
    return (totalCost / len(images), accuracy / len(images)*100)

#Splits a test data set into batches
def splitIntoBatches(data, labels, batchSize):
    numBatches = len(data)//batchSize
    batches = []
    for i in range(numBatches):
        batches.append((data[i*batchSize:(i+1)*batchSize], labels[i*batchSize:(i+1)*batchSize]))
    lastData = data[(numBatches)*batchSize:]
    lastLabels = labels[(numBatches)*batchSize:]
    batches.append((lastData, lastLabels))
    return batches

#Calculates node derivatives for Relu activation
def calculateActivationDerivatives(activation):
    for i, a in enumerate(activation):
        activation[i] = 0 if a<0 else 1
    return activation

#Derivative of the sigmoid function
def reverseSig(a):
    a = sigmoid(a)
    return a * (1-a)

#Calculates node derivatives for Sigmoid activation
def calculateSigmoidDerivatives(activation):
    for i, a in enumerate(activation):
        activation[i] = reverseSig(a)
    return activation

#Updates weight values based on node partial derivatives
def updateWeights(model, learningRate):
    for i,layer in enumerate(model.layers):
        if(i!=0):
            prevLayer = model.layers[i-1]
            gradient = layer.gradient.reshape((len(layer.activations), 1))
            inputs = prevLayer.activations.reshape((1, len(prevLayer.activations)))
            layer.weights -= np.matmul(gradient, inputs) * learningRate
            layer.bias -= learningRate * layer.gradient
        

#Updates each layers gradient values based on gradient of layer above it
def backPropogate(model, target):
    for i in range(len(model.layers)-1, 0, -1):
        layer=model.layers[i]
        if(i==len(model.layers)-1):
            errors = layer.activations - target
        else:
            higherLayer = model.layers[i+1]
            errors = np.matmul(higherLayer.gradient, higherLayer.weights)
        if(layer.activationFun == "sigmoid"):
            derivates = calculateSigmoidDerivatives(np.copy(layer.activations))
        else:
            derivates = calculateActivationDerivatives(np.copy(layer.activations))
        layer.gradient = np.multiply(errors, derivates)

#Trains model by backpropogating through the model and updating its weights
def trainModel(model, data, labels, learningRate = 0.0001, epochs = 3):
    print("Now training...")
    batchSize = 600
    batches = splitIntoBatches(data, labels, batchSize)
    # print(type(batches[0]))
    for epoch in range(epochs):
        print("###################################")
        print("###################################")
        print("###################################")
        print("Epoch :", epoch+1)
        print("###################################")
        print("###################################")
        print("###################################")
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
            print(f"Batch #{i+1},   Accuracy: {round(accuracy,3)}%,   Loss: {round(performance,3)}")
        


#Evaluates model by forward propogating through model for each element of the dataset
#Then calculating the average loss, and accuracy of the model on the whole dataset
def testModel(model, data, labels):
    print("Now testing...")
    totalLoss = 0
    accuracy = 0
    for i,image in enumerate(data):
        label = labels[i]
        label = labels[i]
        cost, accurate = model.evaluate(image, label)
        totalLoss += cost
        accuracy += accurate
    averageLoss = totalLoss/len(data)
    averageAccuracy = accuracy/(i+1)*100
    print(f"Accuracy: {round(averageAccuracy, 3)}%,   Loss: {round(averageLoss,3)}")
    return averageLoss, averageAccuracy