import sys
import argparse
import numpy as np
import nibabel as nib

from scipy.io import loadmat
from pathlib import Path

sys.path.append('.')
from src.data import Data

PATH = '/media/miplab-nas2/Data2/SpinalCord/4_StudentProjects/Daniel/cervical/SPiCiCAPs'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpath', type=str, required=True)
    parser.add_argument('--dpath', type=str, default=PATH)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    # Load cluster components
    clust_path = Path(args.cpath)
    clusts = np.array(loadmat(clust_path)['C'])

    # Load data
    data = Data(PATH, verbose=args.verbose)

    # Convert
    clust_volumes = data.to_volume(clusts)

    # Save
    save_path = clust_path.parent/'clusters.nii'
    nib.save(nib.Nifti1Image(clust_volumes, np.eye(4)), save_path)
