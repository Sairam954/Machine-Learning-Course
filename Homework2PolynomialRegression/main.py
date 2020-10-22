from Homework2PolynomialRegression.polynomialregression.polynomialregression import PolynomialRegression
from Homework2PolynomialRegression.utils.visualize import plot_predicted
import pandas as pd
from tabulate import tabulate

synthetic_dataset_paths = {'Synthetic 1':'./input/synthetic-1.csv',
                           'Synthetic 2':'./input/synthetic-2.csv',
                           'Synthetic 3':'./input/synthetic-3.csv'}
dataset_mse = {}
dataset_theta = {}
order = [1, 2, 4, 7]
order_color = {'1':'green',
                   '2':'orange',
                   '4':'red',
                   '7':'purple'}
for dataset_name in synthetic_dataset_paths.keys():
    dataset = pd.read_csv(synthetic_dataset_paths[dataset_name])
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values
    theta_list = {}
    mse_list = {}
    for degree in order:
        plyregobj = PolynomialRegression(x, y)
        theta_list['order'+str(degree)] = plyregobj.fit(order=degree, epochs=100000, alpha=0.002)
        mse_list['order'+str(degree)] = plyregobj.getmse()
    dataset_mse[dataset_name] = mse_list
    dataset_theta[dataset_name] = theta_list
    plot_predicted(x, y, theta_list.values(), order, order_color, dataset_name)
print('Training Thetha Values for each Dataset for various degree')
print(tabulate(pd.DataFrame.from_dict(dataset_theta).transpose(),headers='keys',tablefmt='grid'))
print('MSE for various dataset for various degree')
print(pd.DataFrame.from_dict(dataset_mse))

