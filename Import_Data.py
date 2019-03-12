# standard data analysis imports
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_iris


def load_shuttle():
    '''
    Loads the shuttle dataset and provides it in digestible format.
    OUTPUT:
        Xtrain: Pandas dataframe containing the training data without class information
        Xtest:  Pandas dataframe containing the test data without class information
    '''
    # column names for the data
    col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'class']

    # loading the training and test set
    Xtrain = pd.read_csv('data/shuttle.trn', header=None, sep=" ", names=col_names)
    Xtest  = pd.read_csv('data/shuttle.tst', header=None, sep=" ", names=col_names)

    return Xtrain.iloc[:,:-1], Xtest.iloc[:,:-1]

def generate_data(n_samples=300, 
                  centers=3,
                  cluster_std=1,
                  random_state=101):
    '''
    Function generating clusters using sklearn make_blobs.
    Input parameters:
        n_samples:      Number of observations created
        centers:        Number of clusters
        cluster_std:    Standard deviation within the clusters
        random_state:   Seed for reproducability
    '''
    X, y_true = make_blobs(n_samples=n_samples, centers=centers,
                       cluster_std=cluster_std, random_state=random_state)
    return X, y_true

def iris_dataset():
    '''
    Load the fisher iris dataset.
    '''
    iris = load_iris()
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df['target'] = iris['target']
    return df.iloc[:,:-1], df.iloc[:,-1]