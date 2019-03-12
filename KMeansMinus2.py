# Author:   Timo Klein
# Date:     2018-12-05
"""
=========================================
K-Means-- Algorithm
=========================================
Implementation of the k-means-- algorithm proposed by Chawla & Gionis.
"""


# necessary imports
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.cluster import AgglomerativeClustering, KMeans

# needed for logging the algorithm's parameters and debugging
import logging
    
# K-Means-- algorithm
class KmeansMinus2(object):
    '''
    K-Means-- Algorithm implementation.
    PARAMETERS:
        n_outliers:                 Predetermined number of outliers to be detected.
        n_centers=3:                number of cluster centers.
        distance='l2':              distance function used for the clustering.
                                    Defaults: Euclidean distance
                                    Can be one of the following:
                                    {‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’,
                                    ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, 
                                    ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, 
                                    ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, 
                                    ‘sokalsneath’, ‘sqeuclidean’, ‘yule’}
        max_iter=30:                Maximum number of iterations performed by the algorithm
        r_seed=101:                 Seed for the random number generator
        initialization='kmpp':      Algorithm used to initialize the centroids.
                                    Possible options: {'random', 'kmpp', 'hierarchy'}.
        verbosity=20:               Level of verbosity in the output log.
                                    Messages of specified level and above are logged.
                                    {0: nothing, 10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR, 50:CRITICAL} 
    '''

    # constructor
    def __init__(self,
                n_outliers, 
                n_centers=7,
                distance_function='l2',
                max_iter=30,
                seed=1,
                initialization='kmpp',
                verbose=20):

        self.n_outliers = n_outliers
        self.n_centers = n_centers
        self.distance_function = distance_function
        self.max_iter = max_iter
        self.initialization = initialization.lower()

        # Seeding
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # set logging level and instantiate a logger
        self.verbose = verbose
        logging.basicConfig(filename='KMeansMinus2.txt')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.verbose)

    # ------------------------------------------------------------------------------------------
    # helper methods. THESE SHOULD NOT BE ACCESSED!

    def __check_input(self, data):
        '''
        Checks whether the input data is of type DataFrame or ndarray.
        Returns a pandas DataFrame.
        Throws an exception datatype is not supported.
        '''
        if isinstance(data, pd.core.frame.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            cols = [f'A{i}' for i in range(1,data.shape[1]+1)]
            return pd.DataFrame(data, columns=cols)
        else:
            # throw exception if data is neither a DataFrame nor a ndarray
            raise TypeError('Datatype is not supported. Please input a pandas DataFrame.')


    def __init_placeholders(self):
        '''
        Initializes placeholders for the fit method.
        '''

        self.outliers = pd.DataFrame(index=range(self.n_outliers), columns=self.data.columns)
        self.n_samples = self.data.shape[0]
        self.dimensions = self.data.shape[1]
        self.centroids = pd.DataFrame(np.nan, index=list(range(self.n_centers)), 
                                                columns=self.data.columns)
        self.data['label'] = np.nan
        self.data['min_distance'] = np.inf
        self.data_removed = None
        self.i = 0
        self.lloss = []

    def __random_init(self):
        '''
        Random centroid initialization
        '''
        r = self.rng.permutation(self.n_samples)[:self.n_centers]
        return self.data.iloc[r,:-2]

    def __kmpp__init(self):
        '''
        Improved centroid initialization according to the k-means++ algorithm.
        Note:   This function alters the original data inplace by
                appending and removing auxiliary columns to it instead of operating on copies.
                This is done to prevent excess memory usage.
        '''

        centroids = self.centroids.copy()
        r = self.rng.choice(self.data.index)
        centroids.iloc[0, :] = self.data.iloc[r,:]

        self.logger.debug(centroids)

        self.data['c_distance'] = pairwise_distances(self.data.iloc[:,:-2], 
                                                        centroids.iloc[[0],:],
                                                        metric=self.distance_function)
        
        sum_squared_dists = self.data['c_distance'].pow(2).sum()
        self.data['probabilities'] = self.data['c_distance'].apply(lambda x: pow(x,2)/sum_squared_dists)

        for k in range(1, self.n_centers):

            r = self.rng.choice(self.data.index, p=self.data['probabilities'])
            centroids.iloc[k, :] = self.data.iloc[r,:]

            _, self.data['c_distance'] = pairwise_distances_argmin_min(self.data.iloc[:,:-4], 
                                                centroids.iloc[:k+1,:],
                                                metric=self.distance_function,
                                                axis=1)
            
            sum_squared_dists = self.data['c_distance'].pow(2).sum()
            self.data['probabilities'] = self.data['c_distance'].apply(lambda x: pow(x,2)/sum_squared_dists)
        

        self.data.drop(['c_distance', 'probabilities'], axis=1, inplace=True)

        return centroids


    def __init_hierarchical(self):
        '''
        Performs centroid initialization according to the hierarchical k-means algorithm.
        Uses single linkage
        '''

        # initialize placeholder for the centroids
        centroids = pd.DataFrame(np.nan, index=list(range(self.n_centers*10)), 
                                                columns=self.data.columns[:-2])

        for j in range(10):

            batch_start = j*self.n_centers
            batch_end = j*self.n_centers+self.n_centers

            km = KMeans(n_clusters=self.n_centers, init='random', n_jobs=-1, n_init=1, precompute_distances=True)
            km.fit(self.data.iloc[:,:-2])
            centroids.iloc[batch_start:batch_end,:] = km.cluster_centers_

            self.rng = np.random.RandomState(np.random.randint(0, 1337))

        self.logger.info(centroids)

        agg_clust = AgglomerativeClustering(n_clusters=self.n_centers, 
                                            affinity=self.distance_function,
                                            linkage='complete')
        agg_clust.fit(centroids)
        centroids['label'] = agg_clust.labels_
        
        centroids = centroids.groupby('label').mean()

        self.rng = np.random.RandomState(self.seed)

        return centroids

    def __init_centroids(self):
        '''
        Function initializing the clustering centroids according to specification.
        'random' initialization samples k centroids uniformly from all observations.
        If not 'random' the k-means++ algorithm will be used to sample the initial centroids.
        '''

        if self.initialization == 'random':
            return self.__random_init()
        elif self.initialization == 'kmpp':
            return self.__kmpp__init()
        else:
            return self.__init_hierarchical()


    def __early_stopping(self, new_outliers, loss):
        ''''
        Returns True if the stopping criteria are met.
        Criterium 1: Set of outliers doesn't change over two iterations.
        Criterium 2: loss change becomes negligible
        '''
        outlier_stop = new_outliers.equals(self.outliers)
        convergence_stop = abs(self.lloss[-1] - loss) < 10e-4
        return (outlier_stop or convergence_stop)

    # ------------------------------------------------------------------------------------------
    # fit kMM to input data
    def fit(self, data):
        '''
        Fits the k-means-- aglorithm to the data given the instance hyperparameters.
        PARAMETERS:
            data:    Input data. Expected to be a pandas Dataframe.
        '''
        

        self.data = self.__check_input(data)
        self.__init_placeholders()
        self.centroids = self.__init_centroids()

        self.logger.info(f'Initial cluster centers: \n{self.centroids}')
        
        while self.i < self.max_iter:

            self.logger.info(f'Iteration number: {self.i}')

            self.data['label'], self.data['min_distance'] = pairwise_distances_argmin_min(self.data.iloc[:,:-2], 
                                                                               self.centroids,
                                                                               metric=self.distance_function,
                                                                               axis=1)
            
            sorted_data = self.data.sort_values('min_distance', 
                                                ascending=False, 
                                                kind='mergesort',
                                                axis=0)
            new_outliers = sorted_data.iloc[:,:-2].head(self.n_outliers)

            self.logger.debug(new_outliers.shape)

            self.sorted_data = sorted_data.drop(sorted_data.index[:self.n_outliers], axis=0)

            self.logger.debug(sorted_data.head(10))

            loss = self.sorted_data['min_distance'].apply(lambda x: pow(x,2)).sum()
            self.centroids = sorted_data.iloc[:,:-1].groupby('label').mean()

            self.logger.info(f'Cluster centers: \n{self.centroids}')

            if self.i > 0:
                if self.__early_stopping(new_outliers, loss):
                    break
            
            self.i += 1

            self.lloss.append(loss)
            self.outliers = new_outliers
            
        # create dataset without outliers after run
        sorted_data = self.data.sort_values('min_distance', 
                                            ascending=False, 
                                            kind='mergesort',
                                            axis=0)
        self.data_removed = sorted_data.drop(sorted_data.index[:self.n_outliers],
                                                axis=0)
        
        self.logger.info('Algorithm completed')
        self.logger.info('-'*100)

    # ------------------------------------------------------------------------------------------
    # various methods for interacting with the label after fitting
    def iterations(self):
        '''
        Returns the number of iterations ran.
        '''
        return self.i
    
    def centers(self):
        '''
        Returns the cluster centers at time of stopping.
        Output is a  pandas DataFrame.
        '''
        return self.centroids.copy()

    def return_outliers(self):
        '''
        Return the detected outliers.
        Output is a  pandas DataFrame.
        '''
        return self.outliers.copy()

    def return_cleaned_data(self):
        '''
        Returns the dataset without outliers.
        Output is a  pandas DataFrame.
        '''
        return self.data_removed.iloc[:,:-2].sort_index(kind='mergesort').copy()
    
    def predict(self):
        '''
        Returns the predicted labels.
        Output is a  pandas DataFrame.
        '''
        return self.data['label'].values.copy()

    def loss(self):
        '''
        Returns a list with loss values for each iteration.
        '''
        return self.lloss.copy()

    def print_params(self):
        '''
        Print the paramters of the algorithm.
        '''
        print(f'KmeansMinus2(n_outliers={self.n_outliers}, n_centers={self.n_centers}, distance_function={self.distance_function}')
        print(f'\tmax_iter={self.max_iter}, r_seed={self.seed}, verbose={logging.getLevelName(self.verbose)})')