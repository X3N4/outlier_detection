# standard data analysis imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import algorithms
from OMeans import OMeans
from KMeansMinus2 import KmeansMinus2
from Agglo_Detect import AggloDetect

# load data
from Import_Data import load_shuttle, generate_data

# basic testing on the shuttle dataset
def shuttle_km():

    k = 3 # number of clusters
    l = 186 # number of outliers

    # load data
    Xtrain, _ = load_shuttle()

    # instantiate and fit the model
    km = KmeansMinus2(l, k, max_iter=100, initialization='hierarchy')
    km.fit(Xtrain)
    print(km.iterations())

    # plot loss
    plt.plot(km.loss())
    plt.show()

def shuttle_agg():

    # load the data
    _, Xtest = load_shuttle()

    # instantiate and fit the algorithm
    agg = AggloDetect()
    agg.fit(Xtest)
    print(agg.stats())

def blobs_km():
    # load data
    X, _ = generate_data()
    plt.scatter(X[:,0], X[:,1], s=50)
    outliers = np.array([[5, -5],
                        [10,-10],
                        [5.5, -6],
                        [-10,10],
                        [-9.5,9],
                        [-15,6],
                        [15,15]])
    X = np.concatenate((X,outliers), axis=0)

    # plot data
    plt.scatter(X[:,0], X[:,1], s=50)
    plt.show()
    # instantiate and fit the model
    km = KmeansMinus2(n_outliers=7, n_centers=3, initialization='hierarchical')
    km.fit(X)

    # extract information
    y_km = km.predict()
    out = km.return_outliers()
    centers = km.centers()

    # plot outliers
    plt.scatter(X[:,0], X[:,1], c=y_km, s=50, cmap='viridis')
    plt.scatter(out['A1'], out['A2'], c='red', s=100, label='Detected outliers')
    plt.scatter(centers['A1'], centers['A2'], c='grey', s=100, label='Cluster centers')
    plt.title('k-means--:Simple outlier detection example', fontsize=14)
    plt.legend()
    # plt.savefig('img/blob_km_example.jpg', dpi=400)
    plt.show()

    plt.plot(km.loss())
    plt.show()


def blobs_agg():
    X, _ = generate_data()
    plt.scatter(X[:,0], X[:,1], s=50)
    outliers = np.array([[5, -5],
                        [10,-10],
                        [5.5, -6],
                        [-10,10],
                        [-9.5,9],
                        [-15,6],
                        [15,15]])
    X = np.concatenate((X,outliers), axis=0)
    plt.show()
    agg = AggloDetect(linkage='complete', distance_function='l1')
    agg.fit(X)
    y_agg = agg.predict()
    out = agg.return_outliers()

    plt.scatter(X[:,0], X[:,1], c=y_agg, s=50, cmap='viridis')
    plt.scatter(out['A1'], out['A2'], c='red', s=100)
    plt.show()

def blobs_om():
    # load data
    X, _ = generate_data(n_samples=1000, centers=3, cluster_std=1, random_state=101)

    outliers = np.array([[5, -5],
                        [10,-10],
                        [5.5, -6],
                        [-10,10],
                        [-9.5,9],
                        [-15,6],
                        [15,15]])
    X = np.concatenate((X,outliers), axis=0)


    fig, axes = plt.subplots(2, 2, figsize=(9,6))
    axes = axes.flatten()

    for i,s in enumerate(range(4,8)):
        cluster = OMeans(n_clusters=3, distance_function='l1',sens=s, 
                thresh='local', detect_global=True, initialization='hierarchy')
        cluster.fit(X)
        y_gms = cluster.predict()
        out = cluster.return_outliers()
        centers = cluster.cluster_centers()
        
        axes[i].scatter(X[:,0], X[:,1], c=y_gms, s=10, cmap='viridis')
        axes[i].scatter(out['A1'], out['A2'], marker='v', c='red', s=100, label='Ausreißer')
        axes[i].scatter(centers['A1'], centers['A2'], c='silver', s=50,label='Clusterzentren')
        axes[i].set_title(f'o-medians mit 3 Clustern und sens={s}',fontsize=14)
        axes[i].legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('img/sens_example.jpg', DPI=400)
    plt.plot(cluster.loss())

def shuttle_om():

    # load data
    Xtrain, Xtest = load_shuttle()

    # instantiate and fit the model
    om = OMeans(n_clusters=7, verbose=10, distance_function='l2',sens=5, 
                thresh='local', detect_global=True, initialization='hierarchy')
    om.fit(Xtrain)

    print(om.cluster_centers().shape[0])
    print(om.return_cleaned_data().shape[0])
    print(om.return_outliers().shape[0])
    print(om.data_stats())
    print(om.outlier_stats())

    # plot loss
    plt.plot(om.loss())
    plt.show()



# -----------------------------------------------------------------------------------------
# function calls for conducting various experiments with datasets

# blobs_km()
# blobs_agg()
# shuttle_agg()
# shuttle_km()
blobs_om()
# shuttle_om()