import scipy.optimize
import numpy as np


def maximum_weight_permutation(M):
    return scipy.optimize.linear_sum_assignment(np.max(M) - M)[1]
