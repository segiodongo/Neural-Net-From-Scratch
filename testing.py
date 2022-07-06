import numpy as np
from numpy import exp

def softmax(vec):
    # vec = vec - max(vec)
    e = exp(vec)
    return e / sum(e)

data = np.array([1,3,2])

print(softmax(data))

print(sum(softmax(data)))