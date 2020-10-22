from Homework1DecisionTrees.decisiontree.ID3decisitiontree import DecisionTree
from Homework1DecisionTrees.utils.discretizedataset import DiscretizeDataset
from pprint import pprint
import pandas as pd
from Homework1DecisionTrees.utils.visualize import plotdecisionBoundary

ddobj = DiscretizeDataset()
dt = DecisionTree()

synthetic_dataset_paths = {'Synthetic 1':'./input/synthetic-1.csv',
                           'Synthetic 2':'./input/synthetic-2.csv',
                           'Synthetic 3':'./input/synthetic-3.csv',
                           'Synthetic 4':'./input/synthetic-4.csv'}

dataset_accuracy = {}
# Synthetic Dataset
for dataset_name in synthetic_dataset_paths.keys():
    dataset = pd.read_csv(synthetic_dataset_paths[dataset_name])

    for col in dataset.columns[:-1]:
        # print(dataset[col].unique())
        if len(dataset[col].unique()) > 2:
            ddobj.equaldistantbin(dataset, col, 4)
        else:
            ddobj.equaldistantbin(dataset, col, 2)


    trainedtree = dt.ID3(dataset, dataset, dataset.columns[:-1], 'class')
    # pprint(trainedtree)
    dataset_accuracy[dataset_name] = dt.test(dataset, trainedtree, 'class')*100
    plotdecisionBoundary(trainedtree,synthetic_dataset_paths[dataset_name],dataset_name)

#pokemon dataset
pokemon_dataset_path = './input/pokemonStats.csv'
pokemon_dataset = pd.read_csv(pokemon_dataset_path)
for col in pokemon_dataset.columns[:-1]:
    # print(dataset[col].unique())
    if len(pokemon_dataset[col].unique()) > 2:
        ddobj.equaldistantbin(pokemon_dataset, col, 4)
    else:
        ddobj.equaldistantbin(pokemon_dataset, col, 2)
pokemon_dt = DecisionTree()
pokemon_trained_tree = pokemon_dt.ID3(pokemon_dataset, pokemon_dataset, pokemon_dataset.columns[:-1], 'Legendary',True)
dataset_accuracy['Pokemon'] = pokemon_dt.test(pokemon_dataset, pokemon_trained_tree, 'Legendary')*100
print("Dataset Accuracy ",dataset_accuracy)