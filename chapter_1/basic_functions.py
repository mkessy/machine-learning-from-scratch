import numpy as np
from numpy import ndarray


def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray
    '''
    return np.power(x, 2)


def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply "Leak ReLU" function to each element in ndarray.
    '''
    return np.maximum(0.2*x, x)


f = 'hi'
print(f)

square('hi')
