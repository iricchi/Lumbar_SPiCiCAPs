import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append('.')
from src.pre_icaps.ica import load_ica
from src.data import GroupData, paths
from src.compare import Comparer
from src.plots.components import plot_component_comparison

plt.rcParams['pdf.fonttype'] = 42


def main(args):
    # Load iCAPS

    path = paths.LUMBAR_PATH
    fpath = Path(args.fpath)/'lumbar'/'ica'/f'K_{args.n_components}'
    fpath.mkdir(parents=True, exist_ok=True)
    template_path = paths.LUMBAR_TEMPLATE

    data = GroupData(path, n_components=args.n_components, 
                template_path=template_path, verbose=args.verbose)
    data.compare_components(template_path)

    ica_path = Path(args.ica_path)/f'K_{args.n_components}'
    ica = load_ica(ica_path)
    ica_results = ica.get_results()

    comp = Comparer(thr1=None)
    comp.compare(data.get_results(), ica_results)

    # Plot similarity
    sim = comp.get_similarity()
    f, ax = plot_component_comparison(sim, fig_path=None)
    ax[0].set_ylabel('iCAPS')
    ax[0].set_xlabel('ICA')
    ax[0].set_xticks([-0.5, args.n_components-0.5])
    ax[0].set_xticklabels([1, args.n_components])
    plt.savefig(fpath/'icaps_similarity.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot similarity between ica and spinal levels
    sims, labs = ica_results.compare_components(data, template_path)
    f, ax = plot_component_comparison(sims, level_labels=labs, fig_path=None)
    ax[0].set_ylabel('ICA components')
    ax[0].set_xlabel('Spinal level')
    plt.savefig(fpath/'ica_spinal_similarity.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('--ica_path', type=str, default='results/lumbar/ica')
    parser.add_argument('-k', '--n_components', type=int, default=9)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    main(args)