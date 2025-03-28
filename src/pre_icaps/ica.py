import pickle as pkl
from nilearn.decomposition import CanICA
from pathlib import Path

from ..data.results import ICAResults

def load_ica(path):
    model = ICA(0)
    path = Path(path)
    if path.is_dir():
        path = path/'model.pkl'
    with open(path, 'rb') as f:
        ica_model = pkl.load(f)
    model._model = ica_model
    model.fitted = True
    return model


class ICA:
    def __init__(self, n_components):
        self.n_components = n_components
        self._model = None
        self.fitted = False
    
    def fit(self, data):
        self._model = CanICA(mask=data.mask_volume, n_components=self.n_components, smoothing_fwhm=None)
        X = data.get_all_epis()
        self._model.fit(X)
        self.fitted = True
    
    def predict(self, data):
        X = data.epi_volume
        return self._model.transform(X)

    def get_components(self, z_score=True):
        if not z_score:
            return self._model.components_
        return (self._model.components_-self._model.components_.mean()) / self._model.components_.std(axis=1, keepdims=True)

    def update_components(self, data):
        if not self.fitted:
            print(f'Model not fitted, fitting on provided data')
            self.fit(data)

        flat_comps = self.get_components(z_score=True)# self._model.components_
        comps = data.to_volume(flat_comps)
        results = ICAResults(comps, flat_comps)
        data.add_results(results)
        return data

    def get_results(self):
        flat_comps = self.get_components(z_score=True)
        results = ICAResults(None, flat_comps)
        return results
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'wb') as f:
            pkl.dump(self._model, f)