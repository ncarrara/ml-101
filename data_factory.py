import numpy.random as rdm
import numpy as np

def linear2D(n, a, b=0, std=1):
    X = []
    Y = []
    for x in range(0, n):
        X.append([x])
        # For OLS (linear regression)
        # The mean of residuals is zero,
        # and we want Homoscedasticity, ie residuals of equal variance
        eps = rdm.normal(0, std)
        y = a * x + b + eps
        Y.append([y])
    return np.array(X), np.array(Y)
