import sys
import argparse
import numpy as np

from pathlib import Path
from scipy.spatial.distance import cdist

sys.path.append('.')
from src.data import GroupData, paths
from src.plots.components import plot_component_comparison_subject


def compute_all_comparisons(all_comps):
    ''' Computes all comparisons between all components and all subjects.
    Args:
        all_comps: (n_subjects, n_components, n_voxels)
    Returns:
        comps: (n_components, n_components, n_subjects*(n_subjects-1)/2)
    '''
    n_subjects, n_comps, n_voxels = all_comps.shape
    flat_comps = all_comps.reshape((n_subjects*n_comps, n_voxels)) # (n_subjects*n_compnents, n_voxels)
    
    comps = 1.0-cdist(flat_comps, flat_comps, metric='dice') # (n_subjects*n_compnents, n_subjects*n_compnents)
    #rcomps = comps.reshape((n_subjects, n_comps, n_subjects*n_comps))
    rcomps = comps.reshape((n_subjects, n_comps, n_subjects, n_comps)) #(n_subjects, n_compnents, n_subjects, n_compnents)
    rcomps = rcomps.transpose((1,3,0,2)) # (n_components, n_components, n_subjects, n_subjects)

    # Get distinct comparisons
    return rcomps.reshape((n_comps, n_comps, n_subjects**2))


def main(args):
    if args.lumbar:
        path = paths.LUMBAR_PATH
        fpath = Path(args.fpath)/'lumbar'/f'K_{args.n_components}'
        template_path = paths.LUMBAR_TEMPLATE
    else:
        path = paths.CERVICAL_PATH
        fpath = Path(args.fpath)/'cervical'/f'K_{args.n_components}'
        template_path = paths.CERVICAL_TEMPLATE

    data = GroupData(path, n_components=args.n_components, 
                     template_path=template_path, subject_acts=True, verbose=args.verbose)
    data.compare_components(template_path)

    all_comps = data.get_subject_components() # (n_subjects, n_components, n_voxels)
    all_comps = all_comps>args.threshold
    comps = compute_all_comparisons(all_comps)

    # Remove ones
    is_one = (comps[0, 0]==1.0)
    comps = comps[:, :, ~is_one]
    plot_component_comparison_subject(comps, data.get_component_label(), fpath/'comps_subjects.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('-k', '--n_components', type=int, default=5)
    parser.add_argument('-t', '--threshold', type=float, default=1.6)
    parser.add_argument('--lumbar', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)