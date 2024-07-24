import os
import random
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    auc,
    precision_recall_curve,
)


def shingle(series, dim):
    """takes a one dimensional series and shingles it into dim dimensions"""
    height = len(series) - dim + 1
    shingled = np.zeros((dim, height))
    for i in range(dim):
        shingled[i] = series[i : i + height]
    return shingled
