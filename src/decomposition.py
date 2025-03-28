import abc
import numpy as np

from sklearn.decomposition import PCA, FastICA


class DecompositionMethod(abc.ABC):
    def fit(self, data):
        self.fit_from_activations(data.activations())
    
    def transform(self, data):
        return self.transform_from_activations(data.activations())

    @abc.abstractmethod
    def get_components(self):
        pass

    @abc.abstractmethod
    def fit_from_activations(self, X):
        pass

    @abc.abstractmethod
    def transform_from_activations(self, X):
        pass


class PCADecomposition(DecompositionMethod):
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self._model = PCA(n_components, **kwargs)

    def fit_from_activations(self, X):
        self._model.fit(X)
    
    def transform_from_activations(self, X):
        return self._model.transform(X)
    
    def get_components(self):
        return self._model.components_


class ICADecomposition(DecompositionMethod):
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self._model = FastICA(n_components, **kwargs)

    def fit_from_activations(self, X):
        self._model.fit(X)
    
    def transform_from_activations(self, X):
        return self._model.transform(X)

    def get_components(self):
        return self._model.components_