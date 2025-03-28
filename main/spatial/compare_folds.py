import sys
import argparse
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm

sys.path.append('.')
from src.data.results import ICAPSResults
from src.data.paths import LUMBAR_PATH
from src.compare import MultipleComparer
from src.plots.compare import plot_sim_mat, plot_single_stability, plot_all_stabilities

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


def read_data(path, n_components):
    # Read main results
    parent_path = Path(path)
    main_path = parent_path/'PAM50_all_Alpha_5_950DOT05'
    data = ICAPSResults(main_path, n_components)

    # Read folds
    fresults = []
    for file in parent_path.iterdir():
        if file.is_dir() and file.name.startswith('PAM50_fold'):
            fold_data = ICAPSResults(file, n_components)
            fresults.append(fold_data)
    return data, fresults


def main_single(args, plot=True):
    main_res, fres = read_data(args.dpath, args.n_components)
    # Compare
    comparer = MultipleComparer(main_res, args.threshold)
    comparer.compare(fres)
    stability = comparer.get_stability()

    # Plot
    if plot:
        fpath = Path(args.fpath)/'lumbar'/f'K_{args.n_components}'/'folds'
        fpath.mkdir(exist_ok=True, parents=True)
        plot_sim_mat(comparer, fpath/'sim_mat.pdf')

        # Plot stability
        plot_single_stability(stability, fpath/'stability.png')
        print(f'Median stability: {np.median(stability):.4f}')

    return stability


def main_all(args):
    all_comps = [5, 6, 7, 8, 9, 10, 11, 15, 20, 25, 30, 35, 40, 50]
    all_sims = []
    for n_components in tqdm(all_comps):
        args.n_components = n_components
        sims = main_single(args, plot=False)
        all_sims.append(sims)
    
    fpath = Path(args.fpath)/'lumbar'
    plot_all_stabilities(all_comps, all_sims, fpath/'stabilities.pdf')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compare folds')
    parser.add_argument('-d', '--dpath', type=str, help='Path to results', default=Path(LUMBAR_PATH).parent)
    parser.add_argument('-f', '--fpath', type=str, help='Path to folds', default='figures')
    parser.add_argument('-k', '--n-components', type=int, help='Number of components', default=None)
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for comparison', default=1.6)
    args = parser.parse_args()

    if args.n_components is not None:
        main_single(args)
    else:
        main_all(args)
