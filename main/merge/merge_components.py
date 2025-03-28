import argparse
import matplotlib.pyplot as plt
from pathlib import Path

import sys

sys.path.append('.')
import src.data.paths as paths
from src.data import GroupData
from src.merger import SupervisedMerger
from src.plots.components import plot_component_comparison, ComponentViewer
from src.plots.merger import plot_merger_iou, plot_tree

plt.rcParams['pdf.fonttype'] = 42

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('--lumbar', action='store_true')
    parser.add_argument('-k', '--n_clusters', type=int, default=5)
    parser.add_argument('-t', '--threshold', type=float, default=3)
    parser.add_argument('-f', '--final', type=int, default=None)
    parser.add_argument('-c', '--criterion', type=str, default='increase', choices=['best', 'increase'])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.lumbar:
        path = paths.LUMBAR_PATH
        fpath = Path(args.fpath)/'lumbar'
        template_path = paths.LUMBAR_TEMPLATE
    else:
        path = paths.CERVICAL_PATH
        fpath = Path(args.fpath)/'cervical'
        template_path = paths.CERVICAL_TEMPLATE
    fpath = fpath/f'K_{args.n_clusters}'/'merge'
    fpath.mkdir(exist_ok=True, parents=True)


    data = GroupData(path, n_components=args.n_clusters, 
                template_path=template_path, verbose=args.verbose)
    comp0, labels0 = data.compare_components(threshold=args.threshold)
    # Plot before merging
    viewer0 = ComponentViewer(data)
    comp_fpath = fpath/'components'/'original'
    comp_fpath.mkdir(exist_ok=True, parents=True)
    viewer0.plot_all(comp_fpath)
    merger = SupervisedMerger(data, criteria=args.criterion, verbose=args.verbose)
    merges, childs = merger.merge(n_final=args.final, threshold=args.threshold)

    # Plot tree
    
    new_icaps = merger.update_components(data.icaps_result)
    comp1, labels1 = new_icaps.compare_components(data, threshold=args.threshold)
    tpath = fpath/'tree.pdf'
    plot_tree(merger, data, fpath=fpath)

    plot_merger_iou(merger, False, fpath)
    plot_merger_iou(merger, True, fpath)
    plot_component_comparison(comp0, labels0, fig_path=fpath/'original_sims.png')
    plot_component_comparison(comp1, labels1, fig_path=fpath/'new_sims.png')

    # Plot after
    viewer1 = ComponentViewer(data)
    comp_fpath = fpath/'components'/'new'
    comp_fpath.mkdir(exist_ok=True, parents=True)
    viewer1.plot_all(comp_fpath)


    