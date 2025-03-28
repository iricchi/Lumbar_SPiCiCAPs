import abc
import numpy as np

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.utils.metric import type_metric, distance_metric

from sklearn.mixture import GaussianMixture


def cosine_distance(x, y):
    return 1 - (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

import warnings
np.warnings = warnings


class ClusterMethod(abc.ABC):
    def fit(self, data):
        self.fit_from_activations(data.activations())
    
    def predict(self, data):
        return self.predict_from_activations(data.activations())

    @abc.abstractmethod
    def fit_from_activations(self, X):
        pass

    @abc.abstractmethod
    def predict_from_activations(self, X):
        pass


class KMeansCluster(ClusterMethod):
    def __init__(self, n_clusters, **kwargs):
        self.n_clusters = n_clusters
        self._model = None
        self._args = kwargs
        self.metric = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)

    def fit_from_activations(self, X):
        #initializer = kmeans_plusplus_initializer(X, self.n_clusters, metric=self.metric)
        initializer = random_center_initializer(X, self.n_clusters)
        init_centers = initializer.initialize()
        self._model = kmeans(X, init_centers, **self._args, metric=self.metric).process()
    
    def predict_from_activations(self, X):
        raise NotImplementedError
        return self._model.predict(X)
    
    def cluster_centers(self):
        cents = np.array(self._model.get_centers())
        return cents


class GMMCluster(ClusterMethod):
    def __init__(self, n_clusters, **kwargs):
        self.n_clusters = n_clusters
        self._model = GaussianMixture(n_clusters, n_init=10, 
                                      init_params='random_from_data', covariance_type='spherical', 
                                      max_iter=100, **kwargs)
    
    def fit_from_activations(self, X):
        self._model.fit(X)
    
    def predict_from_activations(self, X, hard=False):
        if hard:
            return self._model.predict(X)
        return self._model.predict_proba(X)
    
    def cluster_centers(self):
        return self._model.means_
