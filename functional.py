import numpy as np


def relu(input):
    return np.maximum(0, input)


def sigmoid(input):
    return 1 / (1+np.exp(-input))


def softmax(input):
    c = np.max(input)
    exp_input = np.exp(input - c)
    sum_exp_input = np.sum(exp_input)
    ret = exp_input / sum_exp_input

    return ret


def mse_loss(input, target):
    return 0.5 * np.sum((input-target)**2)


def cross_entropy(input, target):
    delta = 1e-7
    return -np.sum(target * np.log(input + delta))


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def SGD(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

