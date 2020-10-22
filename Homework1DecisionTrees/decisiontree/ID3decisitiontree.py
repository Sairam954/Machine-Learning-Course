import numpy as np
import pandas as pd
from pprint import pprint

from Homework1DecisionTrees.utils.discretizedataset import DiscretizeDataset


class DecisionTree:
    no_of_splits = 1
    def __init__(self):
        pass
    def entropy(self,datasetframe,target_column):
        entropy = 0
        target_values = datasetframe[target_column].unique()
        # print("Target Values",target_values)
        for value in target_values:
            prob_of_value = datasetframe[target_column].value_counts()[value]/len(datasetframe[target_column])
            # print('Prob ',prob_of_value)
            entropy += -prob_of_value*np.log2(prob_of_value)
        return entropy
    def IG(self,datasetframe,split_column,target_column):
        dataset_entropy = self.entropy(datasetframe,target_column)
        # vals = datasetframe[split_column].unique()
        # print(vals)
        vals, counts = np.unique(datasetframe[split_column], return_counts=True)
        #
        spilt_column_Entropy = 0
        for i in range(len(vals)):
            # print((datasetframe.where(datasetframe[split_column] == vals[i]).dropna()))
            # print(self.entropy(datasetframe.where(datasetframe[split_column] == vals[i]).dropna(),target_column))
            # spilt_column_Entropy+= (counts[i] / np.sum(counts)) * self.entropy(datasetframe.where(datasetframe[split_column] == vals[i]).dropna()[target_column],target_column)
            #
            spilt_column_Entropy = np.sum([(counts[i] / np.sum(counts)) * self.entropy(datasetframe.where(datasetframe[split_column] == vals[i]).dropna(),target_column) for i in range(len(vals))])
        info_gain = dataset_entropy-spilt_column_Entropy
        return info_gain


    def ID3(self,dataset,originalds, features, target_column="class",depthfixed='False',root_node_class=None):


        if len(dataset[target_column].unique()) <= 1:
              return dataset[target_column].unique()[0]

        elif len(dataset) == 0:

            return np.unique(originalds[target_column])[
                np.argmax(np.unique(originalds[target_column], return_counts=True)[1])]

        elif len(features) == 0:
            return dataset[target_column].value_counts().idxmax()

        else:
            root_node_class = np.unique(dataset[target_column])[ np.argmax(np.unique(dataset[target_column], return_counts=True)[1])]

            feature_ig_values = [self.IG(dataset, feature, target_column) for feature in features]
            best_feature_index = np.argmax(feature_ig_values)
            best_feature = features[best_feature_index]
            # print("Split Feature",best_feature)
            # print("No of splits",self.no_of_splits)
            self.no_of_splits +=1
            tree = {best_feature: {}}

            features = [i for i in features if i != best_feature]

            # print(tree)
            # print(self.tree_depth(tree))


            for value in np.unique(dataset[best_feature]):
                value = value

                sub_dataset = dataset.where(dataset[best_feature] == value).dropna()

                subtree = self.ID3(sub_dataset, dataset,features, target_column,depthfixed,root_node_class)

                tree[best_feature][value] = subtree
                if (depthfixed == True):
                    if (self.no_of_splits > 3):
                        return tree
                # if (self.no_of_splits > 3):
                #     return tree
            return (tree)

    def predict(self,query, tree, default=1):
        for key in list(query.keys()):
            if key in list(tree.keys()):

                try:
                    result = tree[key][query[key]]
                except:
                    return default

                result = tree[key][query[key]]

                if isinstance(result, dict):
                    return self.predict(query, result)

                else:
                    return result

    def test(self,dataset, tree,target_name):

        queries = dataset.iloc[:, :-1].to_dict(orient="records")
        predicted = pd.DataFrame(columns=["predicted"])

        for i in range(len(dataset)):
            predicted.loc[i, "predicted"] = self.predict(queries[i], tree, 1.0)

        print('Accuracy : ', (np.sum(predicted["predicted"] == dataset[target_name]) / len(dataset)) * 100,
              '%')
        return (np.sum(predicted["predicted"] == dataset[target_name]) / len(dataset))

#
# ddobj =  DiscretizeDataset()
# dataset = pd.read_csv('../input/synthetic-1.csv')
#
# for col in dataset.columns[:-1]:
#     # print(dataset[col].unique())
#     if len(dataset[col].unique())> 2:
#         ddobj.equaldistantbin(dataset, col, 4)
#     else:
#         ddobj.equaldistantbin(dataset, col, 2)

    # if (dataset[col].unique()>)
    # ddobj.equaldistantbin(dataset, col, 4)
# dataset = ddobj.equaldistantbin(dataset,'x1',4)
# dataset = ddobj.equaldistantbin(dataset,'x2',4)

# # print(dataset.describe())
# dt = DecisionTree()
# depthlimit = 4
# trainedtree = dt.ID3(dataset,dataset,dataset.columns[:-1],'class',depthlimit)
# pprint(trainedtree)
# dt.test(dataset,trainedtree,'class')
# # # print("Entropy :",dt.entropy(dataset,'class'))
# # print("Info Gain:",dt.IG(dataset,'x1','class'))
