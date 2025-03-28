import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.append('.')
from src.data import Data
from src.decomposition import PCADecomposition, ICADecomposition


PATH = '/media/miplab-nas2/Data2/SpinalCord/4_StudentProjects/Daniel/cervical/SPiCiCAPs'
decomp_methods = {'pca': PCADecomposition, 'ica': ICADecomposition}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=PATH)
    parser.add_argument('--method', type=str, default='ica', choices=['pca', 'ica'])
    parser.add_argument('-k', '--n_components', type=int, default=5)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    data = Data(args.path, verbose=args.verbose)
    model = decomp_methods[args.method](n_components=args.n_components)
    model.fit(data)

    # Save cluster centers to nii files
    centers = data.to_volume(model.get_components())

    out_path = Path('results')/'cervical'/args.method/f'k{args.n_components}'
    out_path.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(centers, data.mask_volume.affine), out_path/f'components.nii')