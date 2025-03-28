import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append('.')
from src.data import GroupData
import src.data.paths as paths
import src.plots.time_courses as tc_plots
import src.plots.components as plots

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 9})
plt.rcParams['pdf.fonttype'] = 42

def main(args):
    if args.lumbar:
        path = paths.LUMBAR_PATH
        fpath = Path(args.fpath)/'lumbar'
        template_path = paths.LUMBAR_TEMPLATE
    else:
        path = paths.CERVICAL_PATH
        fpath = Path(args.fpath)/'cervical'
        template_path = paths.CERVICAL_TEMPLATE
    fpath = fpath/f'K_{args.n_clusters}'/'time_courses'/'couplings'
    fpath.mkdir(parents=True, exist_ok=True)

    data = GroupData(path, n_components=args.n_clusters, 
                     template_path=template_path, verbose=args.verbose)
    data.compare_components()
    labs = data.get_component_label()

    res = data.get_results()
    # Positive
    pc = res.get_couplings()
    f, axs = plots.plot_component_comparison(pc, level_labels=labs, fig_path=None)
    axs[0].set_yticks(np.arange(pc.shape[0])[::-1])
    axs[0].set_yticklabels(labs)
    axs[1].set_ylabel('Jaccard Sim', labelpad=-5)
    axs[0].set_xlabel('iCAPs')
    plt.savefig(fpath/'positive.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Negative
    nc = res.get_couplings(anti=True)
    f, axs = plots.plot_component_comparison(nc, level_labels=labs, fig_path=None)
    axs[1].set_ylabel('Jaccard Sim', labelpad=-5)
    axs[0].set_yticks(np.arange(pc.shape[0]))
    axs[0].set_yticklabels(labs)
    axs[0].set_xlabel('iCAPs')
    plt.savefig(fpath/'negative.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Total
    c = nc + pc
    f, axs = plots.plot_component_comparison(c, level_labels=labs, fig_path=None)
    axs[1].set_ylabel('Jaccard Sim', labelpad=-5)
    axs[0].set_yticks(np.arange(pc.shape[0]))
    axs[0].set_yticklabels(labs)
    axs[0].set_xlabel('iCAPs')
    plt.savefig(fpath/'total.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Per subject
    pc = res.get_couplings(per_subject=True)
    f, axs = plots.plot_component_comparison_subject(pc, labels=labs, fig_path=None)
    axs[0].set_ylabel('Jaccard Sim')
    plt.savefig(fpath/'positive_per_subject.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Negative per subject
    nc = res.get_couplings(anti=True, per_subject=True)
    f, axs = plots.plot_component_comparison_subject(nc, labels=labs, fig_path=None)
    axs[0].set_ylabel('Jaccard Sim')
    plt.savefig(fpath/'negative_per_subject.pdf', dpi=300, bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('-k', '--n_clusters', type=int, default=4)
    parser.add_argument('--lumbar', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    main(args)