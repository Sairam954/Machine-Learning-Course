## Binary Classification  ##

import pandas as pd
import numpy as np
from random import randrange

from Homework3MultiLayerPerceptron.utils.utils import sigmoid, error, cross_entropy, softmax,grad_sigmoid

class FeedForwardNN:
    def __init__(self, x, y):
        self.x = x
        nodes = 256
        self.lr = 0.2
        input_shape = x.shape[1]
        output_shape = y.shape[1]

        self.w1 = np.random.randn(input_shape, nodes)
        self.b1 = np.zeros((1, nodes))
        self.w2 = np.random.randn(nodes, nodes)
        self.b2 = np.zeros((1, nodes))
        self.w3 = np.random.randn(nodes, output_shape)
        self.b3 = np.zeros((1, output_shape))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self, x):
        loss = error(self.a3, self.y)
        x = x + 1
        print('Epoch: %d, Error: ' % x, loss)
        a3_delta = cross_entropy(self.a3, self.y)
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * grad_sigmoid(self.a2)
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * grad_sigmoid(self.a1)

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()

def get_acc(X, Y):
    crct = 0
    for x, y in zip(X, Y):
        s = model.predict(x)
        if s == np.argmax(y):
            crct += 1
    return (crct / X.shape[0]) * 100

