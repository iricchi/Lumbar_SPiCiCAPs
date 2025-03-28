import abc
import numpy as np
import nibabel as nib

from pathlib import Path
from scipy.io import loadmat
from scipy.spatial.distance import cdist

from .paths import SPINAL_LEVELS, LUMBAR_SUBJECTS
from .utils import to_one_hot


def has_time_courses(path):
    ''' Parses the temporal file of the ICAPS results. Looks for a .mat file
        with structure TC_1DOT
    Args:
        path (str): Path to the folder with the ICAPS results.
    Returns:
        str: Name of the folder with the time courses.
    '''
    for file in path.iterdir():
        if file.stem.startswith('tempChar') and file.suffix=='.mat':
            return file


class Results(abc.ABC):
    @abc.abstractmethod
    def get_components(self, flat=False, threshold=None):
        pass
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def label_scores(self):
        return self._label_scores

    @property
    def n_components(self):
        return self._components.shape[-1]

    def flatten(self, volume):
        ''' Masks and flattens volume(s). Inverse of to_volume.
        Args:
            volume (np.ndarray): Array of shape (n_x, n_y, n_z) or (n_x, n_y, n_z, n_components)
        Returns:
            flat_volume (np.ndarray): Array of shape (n_components, n_voxels) or (n_voxels)
        '''
        if len(volume.shape)==3:
            return volume[self.get_mask()].reshape(-1)
        flat_vols = volume[self.get_mask(), :]
        return flat_vols.T

    def get_spinal_levels(self):
        ''' Returns a list of the labels of the distinct spinal levels in the mask
        '''
        return self._spinal_levels

    def get_component_label(self, component=None, score=False):
        if component is None:
            if not score:
                return self.labels
            return self.labels, self.label_scores

        if score:
            return self.labels[component], self.label_scores[component]
        return self.labels[component]

    def load_template(self, template_path):
        self.template = nib.load(template_path).get_fdata().astype(int)

    def print(self, message):
        if self.verbose:
            print(message)
    
    def n_spinal_levels(self, mask=None):
        if mask is not None:
            masked_template = self.template*mask
            return len(np.unique(masked_template))-1
        return len(np.unique(self.template))-1
    
    def compare_components(self, data, template_path=None, threshold=1.6):
        ''' Compares components to a template and reorders them.
        Args:
            template_path (str, optional): Path to the template to compare to. Defaults to cervical levels
            threshold (float, optional): Threshold to consider a component as a spinal cord level.
        Returns:
            np.ndarray: Array of shape (n_components, n_levels) with the similarity scores.
            np.ndarray: Array of shape (n_levels,) with the labels of the spinal cord levels present in the mask.
        '''
        # Load template
        if template_path is not None:
            self.load_template(template_path)
        template_flat = data.flatten(self.template)
        
        labels = SPINAL_LEVELS[np.unique(template_flat)][1:]
        self._spinal_levels = labels
        self._template_oh = to_one_hot(template_flat)
        # Compare components
        if not hasattr(self, '_flat_components'):
            flat_components = data.flatten(self._components)
        else:
            flat_components = self._flat_components
        flat_components = flat_components>threshold
        dists = 1.0-cdist(flat_components, self._template_oh, metric='dice')

        # Reorder
        closest = np.argmax(dists, axis=1)
        max_val = np.max(dists, axis=1)
        sorted_idx = np.lexsort([-max_val, closest])
        self._old2new_order = sorted_idx

        # Sort components
        self._flat_components = flat_components[sorted_idx]
        if self._components is not None:
            self._components = self._components[:, :, :, sorted_idx]
        if hasattr(self, '_time_courses'):
            self._time_courses = self._time_courses[sorted_idx]
        if hasattr(self, '_bin_time_courses'):
            self._bin_time_courses = self._bin_time_courses[sorted_idx]
        if hasattr(self, '_couplings'):
            self._couplings = self._couplings[sorted_idx]
            self._couplings = self._couplings[:, sorted_idx]
        if hasattr(self, '_pcouplings'):
            self._pcouplings = self._pcouplings[sorted_idx]
            self._pcouplings = self._pcouplings[:, sorted_idx]
        if hasattr(self, '_ncouplings'):
            self._ncouplings = self._ncouplings[sorted_idx]
            self._ncouplings = self._ncouplings[:, sorted_idx]
        if hasattr(self, '_subject_acts'):
            self._subject_acts = self._subject_acts[sorted_idx]

        self._labels = labels[closest[sorted_idx]]
        self._label_scores = dists[sorted_idx, closest[sorted_idx]]

        data.print('Components compared to template and reordered')

        return dists[sorted_idx], labels


class ICAResults(Results):
    def __init__(self, components, flat_components, template_path=None, verbose=False):
        self._components = components
        self._flat_components = flat_components
        self.template_path = template_path
        self.verbose = verbose
        if template_path is not None:
            self.load_template(template_path)
    
    def get_components(self, flat=False, threshold=None):
        if flat:
            comps = self._flat_components
        else:
            comps = self._components

        if threshold is None:
            return comps
        return comps>threshold


class ICAPSResults(Results):
    def __init__(self, path, n_components, template_path=None, 
                 subjects=LUMBAR_SUBJECTS, subject_acts=False, 
                 verbose=False):
        self.path = Path(path)
        self._n_components = n_components
        self.template_path = template_path
        self._subjects = subjects
        self._get_subject_acts = subject_acts
        self.verbose = verbose
        if template_path is not None:
            self.load_template(template_path)
        else:
            self.template = None
        
        self.load_components()

    def subject(self, i=None):
        if i is None:
            return self._subjects
        return self._subjects[i]

    @property
    def n_components(self):
        return self._n_components

    def get_mask(self):
        return self._mask

    def set_mask(self, mask):
        self._mask = mask
        self._flat_components = self.flatten(self._components)

    def load_components(self):
        comp_dir = self.path/f'K_{self.n_components}_Dist_cosine_Folds_20'
        cluster_path = comp_dir/'iCAPs_z.nii'
        self._components = nib.load(cluster_path).get_fdata()
        self._mask = self.path/'final_mask.nii'
        self._mask = nib.load(self._mask).get_fdata().astype(bool)
        self._flat_components = self.flatten(self._components)

        # Load consensus
        cons_path = comp_dir/'iCAPs_consensus.mat'
        self.consensus = loadmat(cons_path)['iCAPs_consensus'][:,0]

        # Load time courses
        # Parse temporal file
        tc_file = has_time_courses(comp_dir)
        if tc_file is not None:
            tcs = loadmat(tc_file)['tempChar']
            keys = [dt[0] for dt in eval(str(tcs.dtype))]
            tcs = dict(zip(keys, tcs[0][0]))
            self._time_courses = np.stack(tcs['TC_norm_thes'][0], axis=1)
            self._bin_time_courses = np.stack(tcs['TC_active'][0], axis=1)
            self._couplings = tcs['coupling_jacc']
            self._pcouplings = tcs['coupling_sameSign_jacc']
            self._ncouplings = tcs['coupling_diffSign_jacc']
        else:
            self.print('No temporal file found')
        
        if self._get_subject_acts:
            acts_path = comp_dir/'subjectMaps'
            all_acts = [nib.load(acts_path/f'iCAP_z_{i+1}.nii').get_fdata() for i in range(self._n_components)]
            self._subject_acts = np.stack(all_acts, axis=0)
            print(self._subject_acts.shape)

        self.fs = 1/2.5 # Hz

    def get_subject_acts(self):
        if not hasattr(self, '_subject_acts'):
            raise ValueError('No subject activations found')
        return self._subject_acts

    def get_couplings(self, anti=False, per_subject=False):
        if per_subject:
            if anti:
                return self._ncouplings
            return self._pcouplings
        if anti:
            return np.nanmean(self._ncouplings, axis=-1)
        return np.nanmean(self._pcouplings, axis=-1)

    def get_components(self, flat=False, threshold=None):
        if flat:
            comps = self._flat_components
        else:
            comps = self._components

        if threshold is None:
            return comps
        return comps>threshold
    
    def _update_components(self, new_comps, new_comps_unflat):
        self._components = new_comps_unflat
        self._flat_components = new_comps
        self._n_components = new_comps.shape[0]

    def time_courses(self, binary=False, signed=True):
        ''' Returns computed time courses.
        Args:
            binary (bool, optional): If True, returns binary time courses. Defaults to False.
            signed (bool, optional): If True, returns signed time courses. Defaults to False.
        Returns:
            np.ndarray: array of timecourses of shape (n_components, n_subjects, n_times)
        '''
        if binary:
            if not signed:
                return np.abs(self._bin_time_courses)
            return self._bin_time_courses
        return self._time_courses
