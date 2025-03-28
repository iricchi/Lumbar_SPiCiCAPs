import sys
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import matplotlib as mpl

from pathlib import Path
from shutil import rmtree

sys.path.append('.')
from src.data import AllSubjectsData
from src.data.paths import LUMBAR_SUBJECTS, PAM50_LUMBAR_PATH, LUMBAR_TEMPLATE
from src.plots.components import ComponentViewer, plot_component_comparison
from src.pre_icaps import ICA
from src.pre_icaps.ica import load_ica

PATH = '/media/miplab-nas2/Data2/SpinalCord/Spinal_fMRI_Lumbar/'

mpl.rcParams['pdf.fonttype'] = 42


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_components', type=int, default=10)
    parser.add_argument('-s', '--n_subjects', type=int, default=None)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=1.6)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    fpath = Path('figures')/'lumbar'/'ica'/f'K_{args.n_components}'
    spath = Path('results')/'lumbar'/'ica'/f'K_{args.n_components}'
    fpath.mkdir(exist_ok=True, parents=True)
    
    if args.load:
        data = AllSubjectsData(PATH, only_mask=True, verbose=args.verbose)
        model = load_ica(spath/'model.pkl')
        data = model.update_components(data)
    else:
        data = AllSubjectsData(PATH, n_subjects=args.n_subjects, verbose=args.verbose)
        model = ICA(args.n_components)
        model.fit(data)
        model.save(spath/'model.pkl')
        data = model.update_components(data)

    # Compare components
    comps, labels = data.compare_components(LUMBAR_TEMPLATE, threshold=args.threshold)
    plot_component_comparison(comps, labels, None)
    plt.ylabel('ICA component')
    plt.savefig(fpath/'comparison.pdf', bbox_inches='tight')
    plt.close()
    comp_path = fpath/'components'
    comp_path.mkdir(exist_ok=True)
    viewer = ComponentViewer(data, template_path=PAM50_LUMBAR_PATH)
    viewer.plot_all(comp_path, show_level=False, mask=True)

    # Save components
    volume = nib.Nifti1Image(data.get_components(), data.mask_volume.affine, dtype=np.int64)
    nib.save(volume, spath/'fcomponents.nii.gz')