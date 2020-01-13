import numpy as np


def logic_and(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0

    return 1


def logic_nand(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0

    return 1


def logic_or(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0

    return 1


def logic_xor(x1, x2):
    s1 = logic_nand(x1, x2)
    s2 = logic_or(x1, x2)

    y = logic_and(s1, s2)

    return y

