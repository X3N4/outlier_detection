![O-Means outlier detection example for different sensitivities z](https://imgur.com/o2Fcupx)
# Outlier detection algorithms
This repo contains class based implementations of three outlier detection algorithms.  
Implementations were used for a university project.  

### Outlier detection  Using Clustering Methods [2004, Loureiro et. al]
Uses agglomerative clustering to detect outliers. The hierarchy is cut at a specified number of clusters. All clusters containing less samples than a threshold are considered outliers.  
The original paper can be found [here](https://s3.amazonaws.com/academia.edu.documents/6017200/10.1.1.61.7266.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1552377224&Signature=fzO2QVU%2Bz2igsWE3OT4OBqpUT%2B8%3D&response-content-disposition=inline%3B%20filename%3DOutlier_detection_using_clustering_metho.pdf).
### k-means-- [2013, Chawla & Gionies]
Uses a modified k-means algorithm to perform robust centroid updates with respect to outliers. The points with the largest point to centroid distances are considered outliers.  
The original paper can be found [here](http://pmg.it.usyd.edu.au/outliers.pdf).
### o-medians
Outlier detection algorithm based on k-means# [2017, Olukanmi & Twala]. Uses a k-medians based robust hierarchical initialization [2007, Arai & Barakbah]. Performs k-medians and detects the points which are more than z standard deviations distant from their centroid as outliers. Additionally clusters which are both distant from all other clusters and contain few observations are considered as clusters of outliers.

## Getting Started
Experiments.py contains a number of sample experiments on toy datasets.

### Prerequisites

Python 3.x
* numpy
* pandas
* SciPy
* scikit-learn
* matplotlib

It is recommended to install the requirements through the Anaconda Python distribution.
IMPORTANT: scikit-learn version needs to be >0.20.0, else the function pairwise_distances_argmin_min is bugged.


## Authors

* **Timo Klein** -


## Acknowledgments

* Oscar for being a cool cat and lending his name to the project
