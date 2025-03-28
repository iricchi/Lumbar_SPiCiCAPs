import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append('.')
import src.data.paths as paths
from src.data import GroupData
from src.plots.components import plot_component_comparison, ComponentViewer

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42


def plot_components(data, fpath=Path('.'), **kwargs):
    fpath = fpath/'components'
    fpath.mkdir(parents=True, exist_ok=True)
    plotter = ComponentViewer(data, template_path=paths.PAM50_LUMBAR_PATH)

    for idx in range(data.n_components):
        label, score = data.get_component_label(idx, score=True)
        f = plotter.plot(idx, **kwargs)
        plt.savefig(fpath/f'component_{idx}.pdf', bbox_inches='tight', dpi=300)
        plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('--lumbar', action='store_true')
    parser.add_argument('-k', '--n_clusters', type=int, default=5)
    parser.add_argument('-t', '--threshold', type=float, default=3)
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

    data = GroupData(path, n_components=args.n_clusters, 
                template_path=template_path, verbose=args.verbose)
    sim_matrix, labels = data.compare_components(template_path)


    fpath = fpath/f'K_{args.n_clusters}'
    print(f'Saving figures to {fpath}')
    fpath.mkdir(parents=True, exist_ok=True)
    # plot_components(data, fpath, mask=False)
    # plot_component_comparison(sim_matrix, labels, fpath/'level_similarity.pdf')

    # Plot similarity between levels
    comps = data.icaps_result._flat_components
    from scipy.spatial.distance import cdist
    sims = 1.0-cdist(comps, comps, metric='dice')
    f, ax = plt.subplots(1, 1, figsize=(8, 4))#, gridspec_kw={'width_ratios': [1, 0.1]})
    import seaborn as sns
    cmap = sns.blend_palette(['white', '#0097a7'], as_cmap=True)
    norm = plt.Normalize(0, 1)
    plot = ax.matshow(sims[::-1], cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    #f.colorbar(plot, cax=ax[1])
    plt.savefig(fpath/'comp_similarity.pdf', bbox_inches='tight', dpi=300)
