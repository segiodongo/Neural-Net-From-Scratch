from model import *
from keras.datasets import mnist
from dataReformatting import *
from training import *

#Loading in data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_labels = convertToClassificationMatrix(train_labels)
test_labels = convertToClassificationMatrix(test_labels)
trainData, testData = reformatData(train_images, test_images, 60000, 10000)

model = Model()
layer1 = InputLayer(784)
layer2 = Layer(512, "relu", layer1)
layer3 = Layer(10, "softmax", layer2)

model.addLayer(layer1)
model.addLayer(layer2)
model.addLayer(layer3)
trainModel(model, trainData, train_labels)
