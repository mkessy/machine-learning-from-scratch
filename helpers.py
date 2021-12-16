from typing import Callable, List
import numpy as np
from numpy import ndarray


def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at ever element in the "input_" array
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


Array_Function = Callable[[ndarray], ndarray]

Chain = List[Array_Function]


def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "Chain".
    '''
    assert len(chain) == 2, \
        "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return lambda x: f2(f1(x))

def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input ndarray
    '''

    return 1 / (1 + np.exp(-x))