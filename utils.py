import numpy as np


def shingle(series: np.array, dim: int) -> np.array:
    """
    Takes a one dimensional series and shingles it into dim dimensions.

    Args:
        series (np.array): the input series.
        dim (int): the dimension of the shingled array.

    Returns:
        shingled (np.array): the shingled array.
    """
    height = len(series) - dim + 1
    shingled = np.zeros((dim, height))
    for i in range(dim):
        shingled[i] = series[i : i + height]
    return shingled
