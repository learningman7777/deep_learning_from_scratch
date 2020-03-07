import numpy as np


def relu(input):
    return np.maximum(0, input)


def softmax(input):
    c = np.max(input)
    exp_input = np.exp(input - c)
    sum_exp_input = np.sum(exp_input)
    ret = exp_input / sum_exp_input

    return ret


def mse_loss(input, target):
    return 0.5 * np.sum((input-target)**2)

