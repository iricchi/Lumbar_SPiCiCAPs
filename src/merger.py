import abc
import numpy as np
import nibabel as nib

from sklearn.cluster import AgglomerativeClustering
from .data.utils import to_one_hot

def _compute_union_and_intersection(X1, X2=None):
    ''' 
    Args:
        X1 (np.ndarray): (n_components, n_voxels)
        X2 (np.ndarray): (n_components, n_voxels)
    '''
    if X2 is None:
        X2 = X1

    I = X1 @ X2.T
    U = X1.sum(axis=1).reshape((X1.shape[0], 1)) + X2.sum(axis=1).reshape((1, X2.shape[0])) - I
    return U, I


def _triple_intersection(X1, X2, X3):
    intersec1 = X1.reshape((X1.shape[0], 1, X1.shape[1])) * X2.reshape((1, X2.shape[0], X2.shape[1]))
    I_triple = intersec1 @ X3.T
    return I_triple


# Merge criteria
def best_iou(iou, old_iou):
    return np.unravel_index(np.argmax(iou), iou.shape)

def best_increase(iou, old_iou):
    best_old = np.max(old_iou, axis=1)
    increase = iou-0.5*(best_old[None, :, None]+best_old[:, None, None])
    return np.unravel_index(np.argmax(increase), increase.shape)

spatial_merge_criterium = {'best': best_iou, 'increase': best_increase}


class SupervisedMerger:
    def __init__(self, data, criteria='best', verbose=False):
        self.data = data
        self.n_levels = data.n_spinal_levels
        self.merge_criteria = spatial_merge_criterium[criteria]
        self.verbose = verbose
        self.iou = []

    def print(self, message):
        if self.verbose:
            print(message)

    def merge(self, icaps_result=None, threshold=2.0, n_final=None):
        if icaps_result is None:
            icaps_result = self.data.icaps_result
        if n_final is not None:
            self.n_levels = n_final
        flat_components = self.data.flatten(icaps_result.get_components(threshold=threshold)).astype(int)
        self.flat_levels = to_one_hot(self.data.flatten(icaps_result.template)).astype(int)
        level_sizes = self.flat_levels.sum(axis=1)

        self.n_components = flat_components.shape[0]
        self.n_voxels = flat_components.shape[1]

        # Compute all intersections and unions (shape CxC)
        _, I = _compute_union_and_intersection(flat_components)

        # Compute intersection and unions w.r.t each spinal level (shape CxL)
        U_level, I_level = _compute_union_and_intersection(flat_components, self.flat_levels)
        original_IOU = I_level / U_level

        # Compute triple intersection (component-component-level, shape CxCxL)
        I_triple = _triple_intersection(flat_components, flat_components, self.flat_levels)

        self.iou.append(self._best_iou(original_IOU))
        self.component_childs = [np.array([i]) for i in range(self.n_components)]
        self.sim_level = np.zeros(self.n_components, dtype=int)
        current_n = self.n_components
        self.merges = []
        while current_n > self.n_levels:
            # Find best merge
            new_I = I_level[:, None, :] + I_level[None, :, :] - I_triple
            new_U = U_level[:, None, :] + U_level[None, :, :] - I[:, :, None]+I_triple-level_sizes[None, None, :]
            new_IOU = new_I / new_U
            new_IOU[np.arange(new_IOU.shape[0]), np.arange(new_IOU.shape[0])] = 0

            #best_merge = np.unravel_index(np.argmax(new_IOU), new_IOU.shape)
            best_merge = self.merge_criteria(new_IOU, original_IOU)
            i, j, l = best_merge
            self.sim_level[i] = l
            self.sim_level = np.delete(self.sim_level, j)
            self.print(f'Original: {original_IOU[i, l]:.3f}, {original_IOU[j, l]:.3f}')
    
            assert (i<j) # By construction merge is symmetric therefore i<j
            self.merges.append(best_merge)

            I, U_level, I_level, I_triple, flat_components = self._update_vars(I, U_level, I_level, I_triple, flat_components, i, j)
            original_IOU = I_level / U_level
            self.iou.append(self._best_iou(original_IOU))
            current_n -= 1
            self.print(f'New: {new_IOU[i, j, l]:.3f}, New after: {original_IOU[i, l]:.3f}')
        
        return self.merges, self.component_childs

    def update_components(self, icaps_results):
        ''' Aggregates the components from icaps_results according to the final
            merge.
        '''
        new_comps = np.zeros((self.n_levels, self.n_voxels))
        original_comps = self.data.flatten(icaps_results.get_components(threshold=None))
        for i, childs in enumerate(self.component_childs):
            new_comps[i] = original_comps[childs].max(axis=0)
        new_comps_unflat = self.data.to_volume(new_comps)
        icaps_results._update_components(new_comps, new_comps_unflat)
        return icaps_results

    def get_iou(self):
        return self.iou

    def _best_iou(self, IOU, mean=False):
        # Get best match for each component
        best_iou = np.max(IOU, axis=1)
        if mean:
            return best_iou.mean()
        return best_iou

    def _update_vars(self, I, U_level, I_level, I_triple, flat_components, i, j):
        # Update all variables
        self.component_childs[i] = np.concatenate((self.component_childs[i], self.component_childs[j]))
        self.component_childs.pop(j)
        
        I = np.delete(I, j, axis=0)
        I = np.delete(I, j, axis=1)

        I_level = np.delete(I_level, j, axis=0)
        U_level = np.delete(U_level, j, axis=0)
        I_triple = np.delete(I_triple, j, axis=0)
        I_triple = np.delete(I_triple, j, axis=1)

        # Re-compute intersections and unions
        flat_components[i] = (flat_components[i] + flat_components[j])>0
        flat_components = np.delete(flat_components, j, axis=0)

        # Update row and col of I
        _, I[i] = _compute_union_and_intersection(flat_components[i][None, :], flat_components)
        I[:, i] = I[i]

        # Update row and col of I_level
        U_level[i], I_level[i] = _compute_union_and_intersection(flat_components[i][None, :], self.flat_levels)

        # Update I_triple
        I_triple[i] = _triple_intersection(flat_components[i][None, :], flat_components, self.flat_levels)
        I_triple[:, i] = I_triple[i]

        return I, U_level, I_level, I_triple, flat_components


class UnsupervisedMerger:
    def __init__(self, data, final_n_comps=1, criteria='jaccard', verbose=False):
        self.data = data
        self.merge_criteria = criteria
        self.final_n_comps = final_n_comps
        self.verbose = verbose
        self._model = AgglomerativeClustering(2, metric='precomputed', compute_full_tree=True, 
                                        linkage='complete')
        self.iou = []
    
    def get_component_similarities(self):
        return self._similarities

    def merge(self, icaps_result=None, threshold=2.0):
        if icaps_result is None:
            icaps_result = self.data.icaps_result
        self.old_labels, self.old_sims = icaps_result.get_component_label(score=True)
        flat_components = self.data.flatten(icaps_result.get_components(threshold=threshold)).astype(int)
        n_components = flat_components.shape[0]

        if self.merge_criteria=='jaccard':
            time_courses = icaps_result.time_courses(binary=True).astype(int)
            time_courses = np.abs(time_courses)
        else:
            time_courses = icaps_result.time_courses(binary=False)
        self.n_features = time_courses.shape[1]*time_courses.shape[2]
        time_courses = time_courses.reshape(n_components, self.n_features)


        self.n_components = flat_components.shape[0]
        self.n_voxels = flat_components.shape[1]

        if self.merge_criteria=='jaccard':
            U, I = _compute_union_and_intersection(time_courses)
            self._similarities = I/U
        else:
            self._similarities = np.corrcoef(time_courses)**2
        self._model.fit(1.0-self._similarities)

        self._convert_merges(self._model.children_)

    def update_components(self, icaps_results):
        ''' Aggregates the components from icaps_results according to the final
            merge.
            Args:
                icaps_results: ()
                n_components (int): Desired final number of components.
        '''
        new_comps = np.zeros((self.final_n_comps, self.n_voxels))
        original_comps = self.data.flatten(icaps_results.get_components(threshold=None))

        i = 0
        for childs in self.component_childs:
            if childs is None:
                continue
            new_comps[i] = original_comps[childs].max(axis=0)
            i +=1
        assert(i==(self.final_n_comps))
        new_comps_unflat = self.data.to_volume(new_comps)
        icaps_results._update_components(new_comps, new_comps_unflat)
        return icaps_results
    
    def _convert_merges(self, merges):
        n_merges = self.n_components-self.final_n_comps
        self.component_childs = [np.array([i]) for i in range(self.n_components)]
        self.merges = []
        n_remove = np.zeros(self.n_components, dtype=int)
        renamer = dict(zip(np.arange(self.n_components), np.arange(self.n_components))) # Sklearn when computing a new node it assigns n_childre+1. This remaps it to our format
        for i, merge in enumerate(merges[:n_merges]):
            orig_node1 = renamer[merge[0]]
            orig_node2 = renamer[merge[1]]
            node1 = orig_node1-n_remove[orig_node1]
            node2 = orig_node2-n_remove[orig_node2]
            print(f'Merging {node1} and {node2}')
            if node1 > node2:
                node1, node2 = node2, node1
            self.merges.append((node1, node2, None))
            new_node = self.n_components+i
            assert(not (new_node in renamer))
            assert(node1 < node2)
            renamer[new_node] = orig_node1

            # Perform merge
            self.component_childs[node1] = np.concatenate([self.component_childs[node1], self.component_childs[node2]])
            self.component_childs.pop(node2)
            n_remove[(orig_node2+1):] += 1
            #self.component_childs[node2] = None
        #self.component_childs = [c for c in self.component_childs if c is not None]
