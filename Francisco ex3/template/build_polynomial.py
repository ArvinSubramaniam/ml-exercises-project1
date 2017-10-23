# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    num_samples = len(x)
    ones = np.array([np.ones(num_samples)])
    pol = np.asarray([x**power for power in range(1,degree+1)])
    return np.concatenate((ones, pol), axis=0).T
