# Author:   Timo Klein
# Date:     2018-12-07
"""
=========================================
Agglomerative Clustering for outlier detection
=========================================
Implementation of an outlier detection algorithm based on agglomerative clustering.
The method is described in a paper by Loureiro, Torgo and Soares.
"""


# necessary imports
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# needed for logging the algorithm's parameters and debugging
import logging
    
# K-Means-- algorithm
class AggloDetect(object):
    '''
    Outlier detection algorithm based on agglomerative clustering.
    PARAMETERS:
        max_clust='auto':           Maximum number of clusters 
        cut=10:                     Cluster size at which cutoff occurs.
                                    Clusters with a number of observations larger than cut
                                    are classified as 'normal'.
                                    Clusters with size equal or smaller than cut are classified as
                                    outlier clusters.
        linkage='single':           linkage method used to form the clusters.
                                    Can be one of the following:
                                    {'single', 'complete', 'average', 'ward'}.
                                    IMPORTANT: if linkage is CENTROID, MEDIAN or WARD
                                    the distance function is forced to EUCLIDEAN.
        distance='euclidean':       distance function used for the clustering.
                                    Defaults: Euclidean distance
                                    Can be one of the following:
                                    {‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’,
                                    ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, 
                                    ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, 
                                    ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, 
                                    ‘sokalsneath’, ‘sqeuclidean’, ‘yule’}
        verbosity=20:               Level of verbosity in the output log.
                                    Messages of specified level and above are logged.
                                    {0: nothing, 10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR, 50:CRITICAL} 
    '''

    # constructor
    def __init__(self,
                max_clust='auto',
                cut=5,
                linkage='single',
                distance_function='euclidean',
                verbose=20):
        # max_clust
        self.max_clust = max_clust
        # cut
        self.cut = cut
        # linkage
        self.linkage = linkage.lower()
        # distance function chosen
        self.distance_function = distance_function.lower()
        # set logging level and instantiate a logger
        self.verbose = verbose
        logging.basicConfig(filename='AggloDetect.txt')
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
        # input checks
        if isinstance(data, pd.core.frame.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            # if data is a ndarray we convert into a DataFrame
            cols = [f'A{i}' for i in range(1,data.shape[1]+1)]
            return pd.DataFrame(data, columns=cols)
        else:
            # throw exception if data is neither a DataFrame nor a ndarray
            raise TypeError('Datatype is not supported. Please input a pandas DataFrame.')

    def __init_placeholders(self):
        '''
        Initializes placeholders for the fit method.
        '''
        # number of samples
        self.n_samples = self.data.shape[0]
        # Data point dimensions
        self.dimensions = self.data.shape[1]
        # placeholder for oultiers
        self.outliers = None
        # placeholder for the data without outliers
        self.data_removed = None
        # iterations
        self.i = 0

    def __max_clust(self):
        if isinstance(self.max_clust, str):
            self.max_clust = self.n_samples//10

    def __force_linkage(self):
        if (self.linkage == 'centroid') or (self.linkage == 'median') or (self.linkage=='ward'):
            self.distance_function = 'euclidean'

    def __print_params(self):
        print(f'AggloDetect(max_clust={self.max_clust}, linkage={self.linkage}, distance_function={self.distance_function}')
        print(f'cut={self.cut}, verbose={logging.getLevelName(self.verbose)})')

    # ------------------------------------------------------------------------------------------
    # fit kMM to input data
    def fit(self, data):
        '''
        Fits the Agglo_Detect to the data given the instance hyperparameters.
        PARAMETERS:
            data:    Input data. Expected to be a pandas Dataframe.
        '''
        
        # assign the input data
        self.data = self.__check_input(data)

        # initialize fit method placeholders
        self.__init_placeholders()

        # set max_clust for auto
        self.__max_clust()
        
        # force euclidean distance for centroid, median, ward
        self.__force_linkage()

        # clustering the dataset
        agg_clust = AgglomerativeClustering(n_clusters=self.max_clust, 
                                            affinity=self.distance_function,
                                            linkage='single')
        agg_clust.fit(self.data)
        self.data['label'] = agg_clust.labels_
        
        # adding information about outlier status to each data point
        self.data['outlier'] = self.data['label'].apply(lambda x: 
                                                (self.data['label'].value_counts() <= self.cut)[x])

        # logging statement
        self.logger.info('Number of detected clusters: ' +  str((self.data['label'].value_counts() > self.cut).sum()))


        # determining the outliers
        self.outliers = self.data[self.data['outlier']]

        # logging statement
        self.logger.info(f'Number of detected outliers: {self.outliers.shape[0]}')

        # data without outliers
        self.data_removed = self.data[self.data['outlier'].apply(lambda x: not x)]

        # assert the right dimensions
        assert self.outliers.shape[0] + self.data_removed.shape[0] == self.n_samples
        
        # info statement
        self.logger.info('Algorithm completed')
        self.logger.info('-'*100)

        self.__print_params()
    

    # ------------------------------------------------------------------------------------------
    # various methods for interacting with the class after fitting
    def stats(self):
        '''
        Returns outlier vs non-outlier statistics.
        Output is a  pandas DataFrame.
        '''
        return self.data.groupby('outlier').describe().loc[:, self.data.columns[:-2]].transpose()

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
