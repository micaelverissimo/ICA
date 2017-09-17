from sklearn.metrics import mean_squared_error
from math import sqrt, log10
import scipy.io as sio
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))


def amariError(A, B, squares=False):
    """
    Calculates the performance E1 or E2 as in Oja's book. E2 is square = true, otherwise E1 is compute
    A is the mixture matrix and B is the estimate mixture matrix
    """

    P = np.dot(B,A)

    if squares:
        P = P**2
    else:
        P = np.abs(P)

    max_lines = np.max(P, axis=1)
    max_cols  = np.max(P, axis=0)

    sum_lines = np.sum(np.sum(P/np.tile(max_lines, [P.shape[0], 1]).T, axis=1) -1)
    sum_cols  = np.sum(np.sum(P/np.tile(max_cols , [P.shape[1], 1])  , axis=0) -1)

    return sum_lines + sum_cols