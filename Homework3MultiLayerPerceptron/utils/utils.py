from random import randrange
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grad_sigmoid(x):
    return x * (1 - x)


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(pred, target):
    n_samples = target.shape[0]
    res = pred - target
    return res / n_samples


def error(pred, target):
    n_samples = target.shape[0]
    logp = - np.log(pred[np.arange(n_samples), target.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss

