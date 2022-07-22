Resources:
    3Blue1Brown playlist on neural networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    References for backpropogation algorithm: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    References for softmax algorithm without overflow: https://www.deeplearningbook.org in part 1 chapter 4 pg 79

Instructions:
    Creating Model: use Model() constructor to initialize model;   
        Creating Layers: layer = Layer(numberOfNodes, activationFn, previousLayer)
                    or   layer = InputLayer(numberOfNodes);   
            Note: possible activation functions as of 7/5/2022 are "relu", "sigmoid", and "softmax";   
        Adding Layers: model.addLayer(layer);   
    Training Model: trainModel(model, data, labels, epochs, learningRate);   

