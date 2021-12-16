import numpy as np
from numpy import ndarray
from typing import Callable, List


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


def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at ever element in the "input_" array
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)
