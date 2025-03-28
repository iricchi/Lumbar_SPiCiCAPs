import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.append('.')
from src.data import Data
from src.clustering import KMeansCluster, GMMCluster


PATH = '/media/miplab-nas2/Data2/SpinalCord/4_StudentProjects/Daniel/cervical/SPiCiCAPs'
cluster_methods = {'kmeans': KMeansCluster, 'gmm': GMMCluster}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=PATH)
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'gmm'])
    parser.add_argument('-k', '--n_clusters', type=int, default=5)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    data = Data(args.path, verbose=args.verbose)
    model = cluster_methods[args.method](n_clusters=args.n_clusters)
    model.fit(data)

    # Save cluster centers to nii files
    centers = data.to_volume(model.cluster_centers())
    out_path = Path('results')/'cervical'/args.method/f'k{args.n_clusters}'
    out_path.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(centers, data.mask_volume.affine), out_path/f'clusters.nii')