import numpy as np


def shingle(series, dim):
    """takes a one dimensional series and shingles it into dim dimensions"""
    height = len(series) - dim + 1
    shingled = np.zeros((dim, height))
    for i in range(dim):
        shingled[i] = series[i : i + height]
    return shingled
