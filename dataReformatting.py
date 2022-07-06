import numpy as np

#Reformat Test Images
def reformatData(train_images, test_images, numTrainImages, numTestImages):
    #Reformat Data
    train_images = train_images.reshape((numTrainImages, 784))
    test_images = test_images.reshape((numTestImages, 784))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    return (train_images, test_images)


#Reformat Test Labels
def convertToClassificationMatrix(labels):
    matrix = np.zeros((len(labels), 10), dtype="float32")
    for i in range(len(labels)):
        matrix[i, labels[i]] = 1.0
    return matrix
    