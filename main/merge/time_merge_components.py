
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from shutil import rmtree

import sys

sys.path.append('.')
import src.data.paths as paths
from src.data import GroupData
from src.merger import UnsupervisedMerger
from src.plots.components import plot_component_comparison, ComponentViewer
from src.plots.merger import plot_merger_iou, plot_tree

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('--lumbar', action='store_true')
    parser.add_argument('-k', '--n_clusters', type=int, default=5)
    parser.add_argument('-f', '--final', default=None, type=int)
    parser.add_argument('-t', '--threshold', type=float, default=3)
    parser.add_argument('-c', '--criterion', type=str, default='jaccard', choices=['jaccard', 'pearson'])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.lumbar:
        path = paths.LUMBAR_PATH
        fpath = Path(args.fpath)/'lumbar'
        template_path = paths.LUMBAR_TEMPLATE
        pam50_template_path = paths.PAM50_LUMBAR_PATH
    else:
        path = paths.CERVICAL_PATH
        fpath = Path(args.fpath)/'cervical'
        template_path = paths.CERVICAL_TEMPLATE
        pam50_template_path = paths.PAM50_CERVICAL_PATH
    fpath = fpath/f'K_{args.n_clusters}'/'temporal_merge'
    fpath.mkdir(exist_ok=True, parents=True)


    data = GroupData(path, n_components=args.n_clusters, 
                template_path=template_path, verbose=args.verbose)
    comp0, labels0 = data.compare_components(threshold=args.threshold)
    
    # Plot before merging
    viewer0 = ComponentViewer(data, pam50_template_path)
    comp_fpath = fpath/'components'/'original'
    if comp_fpath.exists():
        rmtree(comp_fpath)
    comp_fpath.mkdir(exist_ok=True, parents=True)
    viewer0.plot_all(comp_fpath, alpha=0.5)

    if args.final is None:
        fcomps = data.n_spinal_levels
    else:
        fcomps = args.final
    merger = UnsupervisedMerger(data, final_n_comps=fcomps, criteria=args.criterion, verbose=args.verbose)
    merger.merge(data.icaps_result, threshold=args.threshold)

    new_icaps = merger.update_components(data.icaps_result)

    comp1, labels1 = new_icaps.compare_components(data, threshold=args.threshold)

    plot_component_comparison(comp0, labels0, fig_path=fpath/'original_sims.png')
    plot_component_comparison(comp1, labels1, fig_path=fpath/'new_sims.png')

    # Plot new comps
    viewer1 = ComponentViewer(data, pam50_template_path)
    comp_fpath = fpath/'components'/'new'
    if comp_fpath.exists():
        rmtree(comp_fpath)
    comp_fpath.mkdir(exist_ok=True, parents=True)
    viewer1.plot_all(comp_fpath, alpha=0.5)

    sims = merger.get_component_similarities()

    plot_component_comparison(comp0, labels0, fig_path=fpath/'original_sims.png')
    plot_component_comparison(comp1, labels1, fig_path=fpath/'new_sims.png')
    plot_tree(merger, data, fpath=fpath/'tree.png')
