import abc
import h5py
import numpy as np
import nibabel as nib

from tqdm.auto import tqdm
from pathlib import Path
from scipy.spatial.distance import cdist

from .results import ICAPSResults
from .utils import c2f_order, to_one_hot
from .paths import LUMBAR_SUBJECTS, SPINAL_LEVELS, LUMBAR_PATH


class Data(abc.ABC):
    @abc.abstractmethod
    def n_voxels(self):
        pass

    @abc.abstractmethod
    def get_mask(self):
        pass
    
    @abc.abstractmethod
    def get_original_shape(self):
        pass

    def to_volume(self, flat_image):
        ''' Creates a 3D volume with the original shape from a flat image.
        Args:
            flat_image (np.ndarray):  Array with the activations of the voxels in the mask, 
                                      can be of shape (n_voxels,) or (n_components, n_voxels).
        '''
        if len(flat_image.shape)==1:
            assert(flat_image.shape[0]==self.n_voxels())
            empty_volume = np.zeros(self.get_original_shape(), dtype=flat_image.dtype)
            empty_volume[self.get_mask()] = flat_image
            return empty_volume
        
        assert(flat_image.shape[1]==self.n_voxels())
        empty_volume = np.zeros(self.get_original_shape()+(flat_image.shape[0],), dtype=flat_image.dtype)
        empty_volume[self.get_mask(), :] = flat_image.T
        return empty_volume
    
    def flatten(self, volume):
        ''' Masks and flattens volume(s). Inverse of to_volume.
        Args:
            volume (np.ndarray): Array of shape (n_x, n_y, n_z) or (n_x, n_y, n_z, n_components)
        Returns:
            flat_volume (np.ndarray): Array of shape (n_components, n_voxels) or (n_voxels)
        '''
        if len(volume.shape)==3:
            assert(volume.shape==self.get_original_shape())
            return volume[self.get_mask()].reshape(-1)
        assert(volume.shape[:3]==self.get_original_shape())
        flat_vols = volume[self.get_mask(), :]
        return flat_vols.T

    def print(self, message):
        if self.verbose:
            print(message)


class GroupData(Data):
    def __init__(self, path, n_components=None, c_order=True, 
                 template_path=None, verbose=False, **kwargs):
        '''
        Wrapper class for the data output of iCAPS.
        Args:
            path (str): Path to the folder with the data.
            n_components (int, optional): If not None, loads preocumpted iCAPS components, otherwise number of components to load.
            c_order (bool, optional): If True, loads the I_sig matrix in C-order, otherwise in F-order. If it is not precomputed it will make the F to C conversion.
            verbose (bool, optional): If True, prints information about the data.
        '''
        self.path = Path(path)
        self.verbose = verbose
        self.c_order = c_order
        self._load_data()
        self.template_path = template_path
        if n_components is not None:
            self.icaps_result = ICAPSResults(self.path, n_components, self.template_path, **kwargs)
        else:
            self.icaps_result = None

    @property
    def template(self):
        return self.icaps_result.template

    @property
    def n_components(self):
        return self.icaps_result.n_components

    @property
    def n_spinal_levels(self):
        return self.icaps_result.n_spinal_levels(self.mask)

    def get_results(self):
        return self.icaps_result

    def change_components(self, n_components):
        self.icaps_result = ICAPSResults(self.path, n_components, self.template_path)

    def get_spinal_levels(self):
        ''' Returns a list of the labels of the distinct spinal levels in the mask
        '''
        return self.icaps_result.get_spinal_levels()

    def get_components(self, flat=False):
        if flat:
            return self.flatten(self.icaps_result._components)
        return self.icaps_result._components

    def get_subject_components(self, subject=None, flat=True):
        comps = self.icaps_result.get_subject_acts()
        comps = np.transpose(comps, (1, 2, 3, 0, 4))
        if subject is not None:
            comps = comps[:, :, :, subject]
        
        return self.flatten(comps) if flat else comps
        

    def get_component_label(self, component=None, score=False):
        return self.icaps_result.get_component_label(component, score)
    
    def compare_components(self, template_path=None, threshold=1.6):
        return self.icaps_result.compare_components(self, template_path, threshold)

    def _load_Isig(self):
        if not self.c_order:
            sig_acts = np.array(h5py.File(self.path /'I_sig.mat', 'r')['I_sig']).T
            return sig_acts
        
        c_order_path = self.path /'I_sig_c_order.npy'
        if c_order_path.exists():
            sig_acts = np.load(c_order_path)
            return sig_acts

        # Otherwise, compute and save
        self.print('Significant actvations not found in C-order, computing them now (might take a while)')
        sig_acts = np.array(h5py.File(self.path /'I_sig.mat', 'r')['I_sig']).T
        sig_acts = c2f_order(sig_acts, self.mask)
        np.save(c_order_path, sig_acts)
        self.print('Significant activations saved to {}'.format(c_order_path))
        return sig_acts

    def _load_data(self):
        self.mask_volume = nib.load(self.path/'final_mask.nii')
        self.mask = self.mask_volume.get_fdata().astype(bool)
        self.original_shape = self.mask.shape

        self.sig_acts = self._load_Isig()

        self.print(f'Number of significant activations: {self.sig_acts.shape[0]}')
        self.print(f'Voxels shape (original): {self.original_shape}, in mask: {self.sig_acts.shape[1]}/{np.prod(self.original_shape)}')

        self.subjects = h5py.File(self.path /'subject_labels.mat', 'r')['subject_labels'][0]
        self.time = h5py.File(self.path /'time_labels.mat', 'r')['time_labels'][0]
        self.print(f'Data loaded from {self.path}')
    
    def activations(self):
        return self.sig_acts
    
    def n_voxels(self):
        return self.sig_acts.shape[1]
    
    def get_mask(self):
        return self.mask
    
    def get_original_shape(self):
        return self.original_shape
    
    def time_courses(self, **kwargs):
        return self.icaps_result.time_courses(**kwargs)


class AllSubjectsData(Data):
    def __init__(self, path, mask_path=Path(LUMBAR_PATH)/'final_mask.nii', 
                 subjects=None, n_subjects=None, only_mask=False,
                 verbose=False):
        self.verbose = verbose
        self.data_path = Path(path)
        self.mask_path = mask_path
        self.only_mask = only_mask
        if n_subjects is not None:
            n_subjects = min(n_subjects, len(LUMBAR_SUBJECTS))
            self.subjects = LUMBAR_SUBJECTS[:n_subjects]
        else:
            self.subjects = LUMBAR_SUBJECTS[subjects] if subjects is not None else LUMBAR_SUBJECTS
        self._load_data()

        self.results = None # Replace by Results object

    def add_results(self, results):
        self.results = results

    @property
    def n_components(self):
        assert(self.results is not None)
        return self.results.n_components

    def get_components(self, flat=False, **kwargs):
        assert(self.results is not None)
        if flat:
            return self.flatten(self.icaps_result._components)
        return self.results._components

    def n_subjects(self):
        return len(self.subjects)

    def get_mask(self):
        return self.mask
    
    def get_original_shape(self):
        return self.original_shape
    
    def n_voxels(self):
        return self._n_voxels

    def get_all_epis(self):
        return self.all_volumes

    def load_template(self, template_path):
        self.template = nib.load(template_path).get_fdata().astype(int)

    def compare_components(self, template_path=None, threshold=1.6):
        assert(self.results is not None)
        return self.results.compare_components(self, template_path, threshold)

    def get_spinal_levels(self):
        ''' Returns a list of the labels of the distinct spinal levels in the mask
        '''
        assert(self.results is not None)
        return self.results.get_spinal_levels()

    def get_component_label(self, component=None, score=False):
        assert(self.results is not None)
        return self.results.get_component_label(component, score)

    def _load_subjects(self):
        for subj in tqdm(self.subjects, desc='Loading volumes'):
            fmri_path = self.data_path/subj/'func'/'sn_mfmri_denoised_all_BP13_2x2x4.nii.gz'
            self.all_volumes.append(nib.load(fmri_path))
            #self.all_epi.append(self.all_volumes[-1].get_fdata())
        
        #self.all_epi = np.stack(self.all_epi, axis=0)
        #print(self.all_epi.shape)

    def _load_data(self):
        self.mask_volume = nib.load(self.mask_path)
        self.mask = self.mask_volume.get_fdata().astype(bool)
        self.original_shape = self.mask.shape
        self._n_voxels = np.sum(self.mask)

        # Load fmri
        self.all_volumes = []
        #self.all_epi = []
        if not self.only_mask:
            self._load_subjects()
        self.print('Data loaded!')


class SubjectData(Data):
    def __init__(self, path, subject):
        self.data_path = Path(path)
        self.subject_name = LUMBAR_SUBJECTS[subject]
        self._load_data()

        self._components = None

    def get_mask(self):
        return self.mask
    
    def get_original_shape(self):
        return self.original_shape

    def n_voxels(self):
        return self._n_voxels

    def _load_data(self):
        mask_path = self.data_path/self.subject_name/'func'/'Segmentation'/'mask_sco.nii.gz'
        self.mask_volume = nib.load(mask_path)
        self.mask = self.mask_volume.get_fdata().astype(bool)
        self.original_shape = self.mask.shape
        self._n_voxels = np.sum(self.mask)

        # Load fmri
        fmri_path = self.data_path/self.subject_name/'func'/'s_mfmri_denoised_all_BP13_2x2x6.nii.gz'#'mfmri_denoised_all_BP13.nii.gz'
        self.epi_volume = nib.load(fmri_path)
        self.epi = self.epi_volume.get_fdata()


    def get_epi(self, flat=False):
        if flat:
            return self.flatten(self.epi)
        return self.epi
    
    def get_components(self, score=None):
        assert(self._components is not None)
        return self._components
    
    @property
    def n_components(self):
        assert(self._components is not None)
        return self._components.shape[-1]

    def get_component_label(self, idx=None, score=None):
        labels = [f'Component {i}' for i in range(self.n_components)]
        if idx is None:
            return labels, [None for _ in range(self.n_components)]
        return labels[idx], None
