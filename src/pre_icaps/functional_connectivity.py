import numpy as np

from sklearn.cluster import AgglomerativeClustering

from ..data.utils import to_one_hot


class FunctionalComponents:
    def __init__(self, n_components):
        self.n_components = n_components
        self._model = AgglomerativeClustering(n_clusters=n_components, metric='correlation', linkage='average')
        self.fitted = False
    
    def fit(self, data):
        X = data.get_epi(True).T # (n_voxels, n_timepoints)
        self._model.fit(X)
        self.fitted = True
    
    def predict(self, data):
        X = data.get_epi(True).T
        assert(X.shape[0]==self._model.labels_.shape[0])
        return self._model.labels_

    def update_components(self, data):
        if not self.fitted:
            print(f'Model not fitted, fitting on provided data')
            self.fit(data)
        
        # Shape (n_voxels)
        flat_comps = self.predict(data)
        flat_comps_oh = to_one_hot(flat_comps, rm_bg=False).astype(int)
        data._flat_components = flat_comps_oh
        comps = data.to_volume(flat_comps_oh)
        data._components = comps.astype(int)
        return data