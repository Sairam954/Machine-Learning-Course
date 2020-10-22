import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style

from Homework2PolynomialRegression.utils.visualize import plot_predicted

style.use('ggplot')

class PolynomialRegression(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def normalise(self, data):
        n_data = (data - np.mean(data)) / (np.max(data) - np.min(data))
        return n_data

    def hypothesis(self, theta, x):

        y_hat = theta[0]
        for i in range(1, len(theta)):
            y_hat = y_hat + theta[i] * x ** i
        return y_hat

    def mse(self, x, y, theta):

        m = len(y)
        y_hat = self.hypothesis(theta, x)
        errors = abs((y_hat - y) ** 2)

        return (1 / (m)) * np.sum(errors)

    def fit(self, order, epochs, alpha):

        d = {}
        d['x' + str(0)] = np.ones([1, len(self.x)])[0]
        for i in np.arange(1, order + 1):
            d['x' + str(i)] = self.normalise(self.x ** (i))

        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        X = np.column_stack(d.values())

        m = len(self.x)
        theta = np.zeros(order + 1)
        mse_list = []
        mser = 0
        for i in range(epochs):
            h = self.hypothesis(theta, self.x)
            errors = h - self.y
            MSE = self.mse(self.x, self.y, theta)
            mse_list.append(MSE)
            theta += -alpha * (1 / m) * np.dot(errors, X)

        self.mser = sum(mse_list) / epochs
        self.epochs = epochs
        self.theta = theta

        return self.theta

    def getmse(self):

        print('Mean Square error = %.4f' % self.mser)
        return self.mser


# Importing the dataset
# dataset = pd.read_csv('../input/synthetic-3.csv')
# x_pts = dataset.iloc[:, 0].values
# y_pts = dataset.iloc[:, -1].values
# # Performing Polynomial Regression
# plyregobj = PolynomialRegression(x_pts, y_pts)
# order = [1,2,4,7]
# theta_list = []
# order_color = {'1':'green',
#                '2':'yellow',
#                '4':'red',
#                '7':'purple'}
# for degree in order:
#
#     theta_list.append(plyregobj.fit(order=degree, epochs=100000, alpha=0.002))
#
#
#
#
#
# plot_predicted(x_pts,y_pts,theta_list,order,order_color)
# Res.printmse()