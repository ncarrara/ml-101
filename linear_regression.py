import numpy as np
from numpy import squeeze
from numpy import transpose as t
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
import timeit

from data_factory import linear2D

NORMAL_EQUATION = "NORMAL"
SGD = "SGD"


def fit_sgd(X, Y, epoch=1000, lr=0.0001):
    w = np.zeros(X.shape[1])
    b = 0
    lr = lr
    for epoch in range(epoch):
        for x, y in zip(X, Y):  # mini batches of size 1
            pred = squeeze(t(w).dot(t(x)) + b)
            err = (y - pred)
            grad_w = (-2 * x) * err
            grad_b = (-2) * err
            w = w - lr * grad_w
            b = b - lr * grad_b

    def f(x):
        return squeeze(t(w).dot(t(x)) + b)

    return f


def fit_normal_equation(X, Y):
    X = np.c_[np.ones((len(X), 1)), X]  # adding ones for bias in affine functions
    tX = t(X)
    w = inv(tX.dot(X)).dot(tX).dot(Y)

    def f(x):
        x = np.c_[np.ones((len(x), 1)), x]  # adding ones for bias in affine functions
        return squeeze(t(w).dot(t(x)))

    return f


if __name__ == "__main__":
    X, Y = linear2D(n=100, a=4, b=-100, std=5)
    starttime = timeit.default_timer()
    Y_NORMAL = fit_normal_equation(X, Y)(X)
    Y_SGD = fit_sgd(X, Y, lr=0.0001,epoch=5000)(X)
    mse_NORMAL = norm(squeeze(Y)-Y_NORMAL,ord=2)
    mse_SGD = norm(squeeze(Y)-Y_SGD,ord=2)
    plt.plot(X, Y, label="data")
    plt.plot(X, Y_NORMAL, label="prediction (normal) mse = {0:.2f}".format(mse_NORMAL))
    plt.plot(X, Y_SGD, label="prediction (sgd) mse={0:.2f}".format(mse_SGD))
    plt.legend()
    plt.show()
