import numpy as np
import pandas as pd

from tqdm.auto import tqdm


def to_one_hot(x, rm_bg=True):
    x = pd.get_dummies(x).values
    if rm_bg:
        return x.T[1:]
    return x.T


def c2f_order(f_array, mask):
    ''' Converts a flat array in F-order to C-order.
    Args:
        f_array (np.ndarray): Array of shape (n_voxels,) or shape (n_components, n_voxels).
        mask (np.ndarray): Array of shape (n_x, n_y, n_z) with boolean values.
    Returns:
        np.ndarray: array of the same shape as f_array but with C-order.
    '''
    flat_mask = mask.flatten(order='F').astype(bool)

    if len(f_array.shape)==1:
        new_array = np.zeros(flat_mask.shape)
        new_array[flat_mask] = f_array
        new_array = new_array.reshape(mask.shape, order='F')
        return new_array[mask]
    
    # Go one-by one to avoid memory errors
    full_new_array = np.zeros((f_array.shape[0], flat_mask.sum()))
    for i, frame in tqdm(enumerate(f_array), desc='Computing C-order'):
        new_array = np.zeros(flat_mask.shape[0])
        new_array[flat_mask] = frame
        new_array = new_array.reshape(mask.shape, order='F')
        full_new_array[i] = new_array[mask]
    return full_new_array