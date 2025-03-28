import sys
import argparse

from pathlib import Path

sys.path.append('.')
import src.data.paths as paths
import src.plots.var as var_plots
from src.data import GroupData
from src.var import VAR

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

def main(args):
    # Load and compare data
    path = paths.LUMBAR_PATH
    fpath = Path(args.fpath)/'lumbar'/f'K_{args.n_clusters}'/'VAR'
    fpath.mkdir(parents=True, exist_ok=True)

    template_path = paths.LUMBAR_TEMPLATE

    data = GroupData(path, n_components=args.n_clusters, 
                template_path=template_path, verbose=args.verbose)
    data.compare_components(template_path)

    model = VAR(order=args.order)
    model.fit(data)
    var_plots.plot_params(data, model, fig_path=fpath/'var_params.pdf')
    var_plots.plot_residuals_corr(data, model, fig_path=fpath/'var_residuals_corr.pdf')
    var_plots.plot_corr(data, fig_path=fpath/'var_corr.pdf')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('-k', '--n_clusters', type=int, default=5)
    parser.add_argument('-d', '--order', default=1, type=int)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
