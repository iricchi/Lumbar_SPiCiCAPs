import numpy as np

from .utils import match
from scipy.spatial.distance import cdist


class Comparer:
    def __init__(self, thr1=1.6, thr2=1.6):
        self._matching = None
        self.thr1 = thr1
        self.thr2 = thr2

    
    def compare(self, results1, results2):
        c1 = results1.get_components(flat=True, threshold=self.thr1)
        c2 = results2.get_components(flat=True, threshold=self.thr2)
        self.compare_from_components(c1, c2)
    
    def compare_from_components(self, c1, c2):
        self._sims = 1.0-cdist(c1, c2, metric='dice')
        self._matching = match(self._sims)

        self.c1 = c1
        # Reorder
        self.c2 = c2[self._matching]
        self._sims = self._sims[:, self._matching]


    def get_matching(self):
        return self._matching
    
    def get_similarity(self):
        return self._sims


class MultipleComparer:
    def __init__(self, base_results=None, threshold=1.6):
        self.base_results = base_results
        self.threshold = threshold
    
    def compare(self, results):
        ''' Compares all results
        Args:
            results: list of Results objects
        '''
        if self.base_results is None:
            self.base_results = results[0]
        self.n_components = self.base_results.n_components

        self.results = results
        # Match and reorder all of them
        self.all_matchings = []
        self.sorted_comps = []
        for r in results:
            r.set_mask(self.base_results.get_mask())
            c1 = self.base_results.get_components(threshold=self.threshold, flat=True)
            c2 = r.get_components(threshold=self.threshold, flat=True)
            sims = 1.0-cdist(c1, c2, metric='dice')
            matching = match(sims)
            self.all_matchings.append(matching)
            self.sorted_comps.append(c2[matching])
        
        # Compute all pairwise similarities
        cat_comps = np.concatenate(self.sorted_comps, axis=0)

        self._all_sims = 1.0-cdist(cat_comps, cat_comps, metric='dice')

    def get_sims(self):
        return self._all_sims

    def n_folds(self):
        return len(self.results)

    def get_folds(self):
        return np.concatenate([(i-1)*np.ones(self.n_components) for i in range(self.n_folds())])

    def get_stability(self, reduce=False):
        # Get secondary diagonal vals
        best_sims = []
        for i in range(self.n_folds()):
            for j in range(i+1, self.n_folds()):
                submat = self._all_sims[i*self.n_components:(i+1)*self.n_components, j*self.n_components:(j+1)*self.n_components]
                best_sims.append(np.diag(submat))
        best_sims = np.concatenate(best_sims)
        if reduce:    
            return np.mean(best_sims)
        return best_sims
    
