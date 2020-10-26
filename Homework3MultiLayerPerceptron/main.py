import numpy as np
import pandas as pd

from Homework3MultiLayerPerceptron.multilayerperceptron.FeedForwardNN import FeedForwardNN, get_acc

train_dataset_path = 'input/mnist_train_0_1.csv'
test_dataset_path =  'input/mnist_test_0_1.csv'

train = np.loadtxt(train_dataset_path, delimiter=',')
test = np.loadtxt(test_dataset_path, delimiter=',')
onehot_target1 = pd.get_dummies(np.transpose(train)[0])
onehot_target2 = pd.get_dummies(np.transpose(test)[0])
# print(onehot_target)
x_train = train[:, 1:]
y_train = onehot_target1

x_val = test[:, 1:]
y_val = onehot_target2

# print("\nTraining accuracy : ", get_acc(x_train, np.array(y_train)))

model = FeedForwardNN(x_train, np.array(y_train))
for xx, yy in zip(x_val, np.array(y_val)):
    s = model.predict(xx)
    # print("\nActual=%d, Predicted=%d" % (np.argmax(yy), s))
model = FeedForwardNN(x_train, np.array(y_train))

epochs = 30
for epoch in range(epochs):
    model.feedforward()
    model.backprop(epoch)
print("\nTest accuracy : ", get_acc(x_val, np.array(y_val)))




