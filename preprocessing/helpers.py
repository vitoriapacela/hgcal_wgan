## By Shah-Rukh

import numpy as np

def _wrap_max(A, B):
    return np.fmod(B + np.fmod(A, B), B)

def _wrap_min_max(A):
    return -np.pi + _wrap_max(A - np.pi, 2*np.pi)


def delta_angle(A,B):
    """
    Computes angle difference between numpy arrays
    """
    return _wrap_min_max(A-B)


def euclidean_two_datasets(A, B):
    """
    Returns euclidean distance between two datasets
    :param A: first dataset in form (N,F), in which N is number of examples in first dataset, and F is number of
    features
    :param B: second dataset in form (M,F)
    :returns: matrix of size (N,M) where each element (i,j) denotes euclidean distance between ith entry in first
    dataset and jth in second dataset
    """

    A = np.array(A)
    B = np.array(B)
    return -2 * A.dot(B.transpose()) + (np.sum(B*B, axis=1)) + (np.sum(A*A, axis=1))[:, np.newaxis]
