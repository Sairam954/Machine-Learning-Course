import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Parameters
from Homework1DecisionTrees.decisiontree.ID3decisitiontree import DecisionTree
from Homework1DecisionTrees.utils.discretizedataset import DiscretizeDataset

def plotdecisionBoundary(trained_tree,dataset_path,dataset_name):
    n_classes = 2
    plot_colors = "rg"
    plot_step = 0.2

    # Load data

    dataset = pd.read_csv(dataset_path)

    X = dataset[['x1','x2']]
    Y = dataset['class']
    y = dataset['class']
    target_names = [0,1]
    # print(X)
    # print(Y)

    dt = DecisionTree()



    # print(Z)
    # plt.subplot(1, 1,1)
    x_min, x_max = dataset['x1'].min()-1,dataset['x1'].max()+1
    y_min, y_max = dataset['x2'].min()-1,dataset['x2'].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    ds = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])

    predicted = pd.DataFrame(columns=["predicted"])
    ds.columns = ['x1','x2']
    ddobj = DiscretizeDataset()
    ddobj.equaldistantbin(ds,'x1',4)
    ddobj.equaldistantbin(ds,'x2',4)
    queries = ds.to_dict(orient="records")

    for i in range(len(ds)):
    # print(dt.predict(queries[i], trainedtree, 1.0))
        predicted.loc[i, "predicted"] = dt.predict(queries[i], trained_tree, 1.0)
    Z = predicted['predicted']
    # print(Z.describe())
    zz = Z.to_numpy().reshape(xx.shape)
    # print(zz.shape)
    # print(xx.shape)
    # print(yy.shape)
    #
    plt.contourf(xx, yy, zz, cmap=plt.cm.RdYlBu)

    dataset_label0 = dataset[dataset['class'].eq(0)]
    dataset_label1 = dataset[dataset['class'].eq(1)]
    plt.scatter(dataset_label0['x1'], dataset_label0['x2'], c='r', edgecolor='black',label='class 0')
    plt.scatter(dataset_label1['x1'], dataset_label1['x2'], c='b', edgecolor='black',label='class_1')
    plt.xlabel('x2')
    plt.ylabel('x1')
    plt.title(dataset_name)

    # for idx in range(len(y)):
    # # print(idx)
    # # print(X['x1'][idx],X['x2'][idx])
    #     if(dataset['class'][idx]==0):
    #         color = 'r'
    #         label = 0
    #
    #     else:
    #         color = 'b'
    #         label = 1
    #     plt.scatter(X['x1'][idx], X['x2'][idx],c=color,edgecolor='black')
    # #
    # for i, color in zip(range(n_classes), plot_colors):
    #         idx = np.where(y == i)
    #         for id in idx:
    #             print("index",id)
    #         #     plt.scatter(X[id,0],X[id,1],c=color)
    #         # # print(idx)
    #         # print(X[idx, 0])
    #            # plt.scatter(X[idx, 0], X[idx, 1], c=color, label=target_names[i],
    #         #             cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    #
    #

    plt.legend()
    plt.show()


# Z = Z.to_numpy().reshape(xx.shape)
# print(Z.reshape(xx.shape))

#     clf = DecisionTreeClassifier().fit(X, y)
#
#     # Plot the decision boundary
#     plt.subplot(2, 3, pairidx + 1)
#
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                          np.arange(y_min, y_max, plot_step))
#
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
#     plt.xlabel(iris.feature_names[pair[0]])
#     plt.ylabel(iris.feature_names[pair[1]])
#     plt.axis("tight")
#
#     # Plot the training points
#     for i, color in zip(range(n_classes), plot_colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
#                     cmap=plt.cm.Paired)
#
#     plt.axis("tight")
#
# plt.suptitle("Decision surface of a decision tree using paired features")
# plt.legend()
# plt.show()



# sns.set(style="white", color_codes=True)
#
# synthetic_dataset1 = pd.read_csv("../input/synthetic-1.csv")
# synthetic_dataset2 = pd.read_csv("../input/synthetic-2.csv")
# synthetic_dataset3 = pd.read_csv("../input/synthetic-3.csv")
# synthetic_dataset4 = pd.read_csv("../input/synthetic-4.csv")
#
# # print("class distribution of dataset 1")
# # print(synthetic_dataset1['class'].value_counts())
# #
# # print("class distribution of dataset 2")
# # print(synthetic_dataset2['class'].value_counts())
# #
# # print("class distribution of dataset 3")
# # print(synthetic_dataset3['class'].value_counts())
# #
# # print("class distribution of dataset 4")
# # print(synthetic_dataset4['class'].value_counts())
# sns.FacetGrid(synthetic_dataset1, hue= "class").map(plt.scatter,"x1","x2").add_legend()
#
# sns.FacetGrid(synthetic_dataset2, hue= "class").map(plt.scatter,"x1","x2").add_legend()
#
# sns.FacetGrid(synthetic_dataset3, hue= "class").map(plt.scatter,"x1","x2").add_legend()
#
# sns.FacetGrid(synthetic_dataset4, hue= "class").map(plt.scatter,"x1","x2").add_legend()
# plt.show()
