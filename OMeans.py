# Author:   Timo Klein
# Date:     2018-12-17
"""
=========================================
O-Means Algorithm
=========================================
O-Means is an algorithm for outlier detection based on k-means# by Olukanmi & Twala.
"""


# necessary imports
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.cluster import KMeans, AgglomerativeClustering

# needed for logging the algorithm's parameters and debugging
import logging
    
# K-Means-- algorithm
class OMeans(object):
    '''
    K-Means-- Algorithm implementation.
    PARAMETERS:
        n_clusters=3:               number of cluster centers.
        distance='l1':              distance function used for the clustering.
                                    Defaults: Euclidean distance
                                    Can be one of the following:
                                    {‘cityblock’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, 
                                    ‘seuclidean’, ‘sqeuclidean’}
                                    If distance is one of {'cityblock', 'manhattan', 'l1'}
                                    the algorithm performed is k-medians.
        sens=5:                     Number of standard deviations from class mean
                                    after which a point is defined as outlier.
        thresh='local':            Determines if threshold value is 'global' (=same for all clusters)
                                    or 'local' (dependent on cluster).
        detect_global=True:         detetermines if OMeans detects clusters of outliers or not.
                                    boolean value.
        max_iter=30:                Maximum number of iterations performed by the algorithm
        r_seed=None:                Seed for the random number generator
        initialization='hierarchy':    Algorithm used to initialize the centroids.
                                    Possible options: {'random', 'kmpp','hierarchy'}.
        verbose=20:                 Level of verbosity in the output log.
                                    Messages of specified level and above are logged.
                                    {0: nothing, 10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR, 50:CRITICAL} 
    '''

    # constructor
    def __init__(self,
                n_clusters,
                distance_function='l1',
                sens=5,
                thresh='local',
                detect_global=True,
                max_iter=50,
                seed=101,
                initialization='hierarchy',
                verbose=20):
        
        self.n_clusters = n_clusters
        self.distance_function = distance_function
        self.sens = sens
        self.thresh = thresh.lower()
        self.detect_global = detect_global
        self.max_iter = max_iter
        self.epsilon = 10e-4 # accuracy paramter

        # Seeding
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.initialization = initialization.lower()

        # set logging level and instantiate a logger
        self.verbose = verbose
        logging.basicConfig(filename='OM.txt')
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
            cols = [f'A{i}' for i in range(1,data.shape[1]+1)]
            return pd.DataFrame(data, columns=cols)
        else:
            raise TypeError('Datatype is not supported. Please input a pandas DataFrame.')


    def __check_distance(self):
        '''
        Performs an input check and initializes the distance function in unified manner.
        This helps with later conditional statements.
        '''
        if self.distance_function == 'manhattan' or self.distance_function == 'cityblock':
            self.distance_function = 'l1'
        elif self.distance_function == 'seuclidean':
            self.distance_function = 'sqeuclidean'
        else:
            self.distance_function = 'l2'

    def __init_variables(self):
        '''
        Initializes variables and placeholders for the fit method.
        '''
        self.outliers = None

        self.n_samples = self.data.shape[0]
        self.dimensions = self.data.shape[1]
        self.data['label'] = np.nan
        self.data['min_distance'] = np.inf

        self.centroids = pd.DataFrame(np.nan, index=list(range(self.n_clusters)), 
                                        columns=self.data.columns[:self.dimensions])

        self.data_removed = None
        self.i = 0
        self.lloss = []
        self.T = None


    def __centroid_update(self, data):
        '''
        Calculation of the centroid update with respect to the distance function.
        OUTPUT:
            centroids:  n_clusters new centroids according to the dataset partition
        '''
        if self.distance_function == 'l1':
            centroids = data.groupby('label').median()
        else:
            centroids = data.groupby('label').mean()
        
        return centroids

    def __kmedians_plain(self):
        '''
        Plain k-means implementation used to calculate centers for 
        hierarchical.
        '''
        centers = self.__random_init(self.data, self.n_clusters)
    
        while True:
            self.data['label'], self.data['min_distance'] = pairwise_distances_argmin_min(self.data.iloc[:,:-2], 
                                                                               centers,
                                                                               metric='l1',
                                                                               axis=1)
            
            new_centers = self.__centroid_update(self.data.iloc[:,:-1])

            if new_centers.equals(centers):
                    break

            centers = new_centers

        # perform cleanup
        self.data['label'] = np.nan
        self.data['min_distance'] = np.inf

        return centers

    def __random_init(self, data, n_clusters):
        '''
        Random centroid initialization
        INPUTS:
            n_clusters:         number of cluster centers
        OUTPUTS:
            initial centroids:  n_cluster datapoints, selected uniform at random
                                from the dataset
        '''
        r = self.rng.permutation(data.shape[0])[:n_clusters]
        return data.iloc[r,:-2]

    def __kmpp__init(self):
        '''
        Improved centroid initialization according to the k-means++ algorithm.
        Note:   This function alters the original data inplace by
                appending and removing auxiliary columns to it instead of operating on copies.
                This is done to prevent excess memory usage.
        OUTPUTS:
            centroids:          n_cluster datapoints initialized according to 
                                the k-means++ algorithm
        '''

        # initialize first center randomly
        centroids = self.centroids.copy()
        r = self.rng.choice(self.data.index)
        centroids.iloc[0, :] = self.data.iloc[r,:]

        self.logger.debug(centroids)

        self.data['c_distance'] = pairwise_distances(self.data.iloc[:,:-2], 
                                                        centroids.iloc[[0],:],
                                                        metric=self.distance_function)
        

        # assign probabilities to each observation
        sum_squared_dists = np.square(self.data['c_distance']).sum()
        self.data['probabilities'] = np.square(self.data['c_distance'])/sum_squared_dists

        message = 'Incorrect probability calculation occured during k-means++'
        assert (abs(1 - self.data['probabilities'].sum()) < self.epsilon), message

        # continue adding new centroids and recalculate distances & probabilities
        for k in range(1, self.n_clusters):
            r = self.rng.choice(self.data.index, p=self.data['probabilities'])
            centroids.iloc[k, :] = self.data.iloc[r,:]

            _, self.data['c_distance'] = pairwise_distances_argmin_min(self.data.iloc[:,:-4], 
                                                                        centroids.iloc[:k+1,:],
                                                                        metric=self.distance_function,
                                                                        axis=1)
            
            sum_squared_dists = np.square(self.data['c_distance']).sum()
            self.data['probabilities'] = np.square(self.data['c_distance'])/sum_squared_dists

        # remove the helper columns from the data
        self.data.drop(['c_distance', 'probabilities'], axis=1, inplace=True)

        return centroids

    def __hierarchical_init(self):
        '''
        Performs centroid initialization according to the hierarchical k-means algorithm.
        Differs from hierarchical k-means in that k-medians is used to obtain the initial centers.
        Complete linkage is used to perform agglomerative clustering on these centers.
        OUTPUT:
            centroids:          n_cluster points initialized with hierarchical initalization
        '''
        trials = 10

        # initialize placeholder for the centroids
        centroids = pd.DataFrame(np.nan, index=list(range(self.n_clusters*trials)), 
                                                columns=self.data.columns[:-2])

        for j in range(trials):

            batch_start = j*self.n_clusters
            batch_end = j*self.n_clusters+self.n_clusters

            # ensure correct number of centers (kmedians can dissolve cluster centers)
            while True:
                candidate_centers = self.__kmedians_plain().values
                if candidate_centers.shape[0] == self.n_clusters:
                    break

            centroids.iloc[batch_start:batch_end,:] = candidate_centers

            self.rng = np.random.RandomState(np.random.randint(0, 1337))

        # debugging statement
        self.logger.info(centroids)

        # assign labels to all found centers based on hierarchical clustering
        agg_clust = AgglomerativeClustering(n_clusters=self.n_clusters, 
                                            affinity=self.distance_function,
                                            linkage='complete')
        agg_clust.fit(centroids)
        centroids['label'] = agg_clust.labels_
        
        # compute final centroids
        centroids = centroids.groupby('label').median()

        # reset the random state
        self.rng = np.random.RandomState(self.seed)

        return centroids

    def __init_centroids(self):
        '''
        Checks the value of self.initialization and initializes centroids accordingly.
        The default initialization method is hierarchical k-means.
        '''

        if self.initialization == 'random':
            return self.__random_init(self.data, self.n_clusters)
        elif self.initialization == 'kmpp':
            return self.__kmpp__init()
        elif self.initialization == 'hierarchy':
            return self.__hierarchical_init()
            
            

    def __local_outliers(self):
        '''
        Calculates local outliers depending on the value of self.thresh.
        self.thresh == 'local':     Estimates the standard deviation of the point to centroid distance
                                    for each cluster seperately through the MAD.
                                    All points with a distance to their centroid
                                    greater than sens*STD are considered outliers.
        self.thresh == 'global':    Estimates the standard deviation of the point to centroid distance
                                    for the whole dataset through the MAD.
                                    All points with a distance to their centroid
                                    greater than sens*STD are considered outliers.
                                    IMPORTANT: This setting can remove clusters if all their points are
                                    considered local outliers!
                                    This is the default value.
        OUTPUT:
            outliers:           points in the dataset which are further than T
                                away from their respective cluster center.
        '''
        if self.thresh == 'local':
            medians = dict(self.data.groupby('label').median()['min_distance'])
            self.data['MAD'] = self.data.apply(lambda x: abs(x['min_distance'] - medians[int(x['label'])]), axis=1)
            self.T = dict(self.sens*(1.4826*self.data.groupby('label').median()['MAD']))

            self.logger.info(f'Threshold values: {self.T}')

            # housekeeping
            self.data.drop('MAD', axis=1, inplace=True)

            # outlier value (boolean) for each observation
            out = self.data.apply(lambda x: x['min_distance'] >= self.T[int(x['label'])], axis=1)
        
        else:
            medians = dict(self.data.groupby('label').median()['min_distance'])
            self.data['MAD'] = self.data.apply(lambda x: abs(x['min_distance'] - medians[int(x['label'])]), axis=1)
            self.T = self.sens*(1.4826*self.data['MAD'].median())

            self.logger.info(f'Threshold value: {self.T}')

            # housekeeping
            self.data.drop('MAD', axis=1, inplace=True)

            # outlier value (boolean) for each observation
            out = self.data.apply(lambda x: x['min_distance'] >= self.T, axis=1)            

        return self.data[out].iloc[:,:-1]
    

    def __global_outliers(self):
        '''
        Decides whether a cluster is a global outlier. Only clusters which are both
        distant from other clusters and have low population sizes are considered outliers.
        OUTPUT:
            global outliers:        list of labels considered global outlier
        '''
        dist_mat = pairwise_distances(self.centroids, metric=self.distance_function)
        ICD = np.einsum('ij->i', dist_mat)/(self.centroids.shape[0] - 1)
        distant = ICD > np.average(ICD) + np.std(ICD)
            
        counts = self.data_removed['label'].value_counts()
        MAD = (abs(counts - counts.median())).median()
        population = counts < counts.median() - MAD

        self.logger.debug(f'Distant: \n{distant}')
        self.logger.debug(f'Global outlier counts: \n{counts}')
        self.logger.debug(f'Global outlier sparse: \n{population}')

        global_outs = []
        for i in population.index:
            if distant[i] and population[i]:
                global_outs.append(i)

        if len(global_outs) != 0:
            self.logger.info(f'Removed global outliers: {global_outs}')

        return self.data_removed[self.data_removed['label'].isin(global_outs)]

    def __early_stopping(self, new_outliers, loss):
            ''''
            Returns True if the stopping criteria are met.
            Criterion 1: Set of outliers doesn't change over two iterations.
            Criterion 2: loss change becomes negligible
            '''

            outlier_stop = new_outliers.equals(self.outliers)
            convergence_stop = abs(self.lloss[-1] - loss) < self.epsilon
            return (outlier_stop or convergence_stop)
    
    def __calculate_loss(self):
        '''
        Calculation of the Loss depending on the distance function.
        If the distance used is 'l1', the loss is the sum of distances.
        For 'euclidean' and 'sqeuclidean' the loss is the sum of squared distances.
        OUTPUT:
            loss:   Value of the loss function for a given partition.
        '''
        if self.distance_function == 'l1':
            loss = self.data_removed['min_distance'].sum()
        else:
            loss = np.square(self.data_removed['min_distance']).sum()
        
        return loss

    # ------------------------------------------------------------------------------------------
    
    def fit(self, data):
        '''
        Fits the k-means-- aglorithm to the data given the instance hyperparameters.
        INPUTS:
            data:    Input data. Expected to be a pandas Dataframe.
        '''
        

        self.data = self.__check_input(data)
        self.__init_variables()
        self.centroids = self.__init_centroids()

        self.logger.info(f'Initial cluster centers: \n{self.centroids}')
        
        # training loop
        while self.i < self.max_iter:

            self.logger.info(f'Iteration number: {self.i}')

            self.data['label'], self.data['min_distance'] = pairwise_distances_argmin_min(self.data.iloc[:,:-2], 
                                                                                            self.centroids,
                                                                                            metric='l2',
                                                                                            axis=1)

            
            information = pd.DataFrame({'label counts': self.data['label'].value_counts(),
                            'intrac distance': self.data.groupby('label')['min_distance'].mean(),
                            'T': self.T})
            self.logger.info(f"Stats before outlier removal:\n{information}")

            local_outliers = self.__local_outliers()
            self.data_removed = self.data.drop(local_outliers.index, axis=0)
            new_outliers = local_outliers
            if self.detect_global and self.centroids.shape[0] > 1:
                global_outliers = self.__global_outliers()
                self.data_removed = self.data_removed.drop(global_outliers.index, axis=0)
                new_outliers = pd.concat([local_outliers, global_outliers ], axis=0, verify_integrity=True, sort=False)

            self.logger.debug(new_outliers.shape)

            loss = np.square(self.data_removed['min_distance']).sum()

            if self.i > 0:
                if self.__early_stopping(new_outliers, loss):
                    break

            self.centroids = self.__centroid_update(self.data_removed.iloc[:,:-1])

            self.lloss.append(loss)
            self.outliers = new_outliers

            self.logger.info(f'Cluster centers: \n{self.centroids}')

            self.i += 1

        self.logger.info('Algorithm completed')
        self.logger.info('-'*100)

    # ------------------------------------------------------------------------------------------
    # various methods for interacting with the label after fitting
    def iterations(self):
        '''
        Returns the number of iterations.
        '''
        return self.i
    
    def cluster_centers(self):
        '''
        Returns the cluster centers at time of stopping in a DataFrame.
        '''
        return self.centroids.copy()

    def data_stats(self):
        '''
        Returns cluster statistics for clean data.
        The output is a DataFrame
        '''
        return self.data_removed.iloc[:,:-1].groupby('label').describe().transpose()

    def outlier_stats(self):
        '''
        Returns cluster statistics for outliers.
        The output is a DataFrame
        '''
        return self.outliers.groupby('label').describe().transpose()

    def return_outliers(self):
        '''
        Return the detected outliers as DataFrame.
        '''
        return self.outliers.copy()

    def return_cleaned_data(self):
        '''
        Returns the dataset without outliers as DataFrame.
        '''
        return self.data_removed.iloc[:,:-2].sort_index(kind='mergesort').copy()
    
    def predict(self):
        '''
        Returns the predicted labels as DataFrame.
        '''
        return self.data['label'].copy()

    def loss(self):
        '''
        Returns a list with loss values for each iteration.
        '''
        return self.lloss.copy()

    def print_params(self):
        '''
        Prints all parameters of the algorithm.
        '''
        message = (
            f'OMeans(n_clusters={self.n_clusters}, '
            f'distance_function={self.distance_function}, '
            f'sens={self.sens}, '
            f'thresh={self.thresh}, '
            f'detect_global={self.detect_global}, '
            f'\n\tmax_iter={self.max_iter}, '
            f'r_seed={self.seed}, '
            f'verbose={logging.getLevelName(self.verbose)}, '
            f'initialization={self.initialization})'
            )
        print(message)