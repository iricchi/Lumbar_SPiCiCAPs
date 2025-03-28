import sys
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
from shutil import rmtree

sys.path.append('.')
from src.data import SubjectData
from src.pre_icaps.functional_connectivity import FunctionalComponents
from src.data.paths import LUMBAR_SUBJECTS
from src.plots.components import ComponentViewer

PATH = '/media/miplab-nas2/Data2/SpinalCord/Spinal_fMRI_Lumbar/'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=int, default=0)
    parser.add_argument('-n', '--n_components', type=int, default=10)
    args = parser.parse_args()
    fpath = Path('figures')/'lumbar'/LUMBAR_SUBJECTS[args.subject]/'fc'
    fpath.mkdir(exist_ok=True, parents=True)

    data = SubjectData(PATH, args.subject)
    model = FunctionalComponents(args.n_components)
    data = model.update_components(data)

    comp_path = fpath/'components'
    comp_path.mkdir(exist_ok=True)

    template_path = Path('templ')
    viewer = ComponentViewer(data, template_path=template_path)
    viewer.plot_all(comp_path, show_level=False, mask=True)

    # Save components
    volume = nib.Nifti1Image(data.get_components(), data.mask_volume.affine, dtype=np.int64)
    rpath = Path('results')/'cervical'/LUMBAR_SUBJECTS[args.subject]
    rpath.mkdir(exist_ok=True, parents=True)
    nib.save(volume, rpath/'fcomponents.nii.gz')