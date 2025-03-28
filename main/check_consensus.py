import argparse
import sys

from pathlib import Path

sys.path.append('.')
from src.data import GroupData
from src.data.paths import LUMBAR_PATH
from src.plots.components import plot_consensus


cPATH = '/media/miplab-nas2/Data2/SpinalCord/4_StudentProjects/Daniel/cervical/SPiCiCAPs'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lumbar', action='store_true')
    args = parser.parse_args()

    if args.lumbar:
        path = Path(LUMBAR_PATH)
        fpath = Path('figures')/'lumbar'
    else:
        path = Path(cPATH)
        fpath = Path('figures')/'cervical'
    fpath.mkdir(parents=True, exist_ok=True)

    # Parse ks from the folders
    ks = []
    for k in path.glob('K_*'):
        if k.is_dir():
            ks.append(int(k.name.split('_')[1]))
    data = GroupData(path, n_components=None, c_order=False, verbose=True)
    plot_consensus(data, ks, fpath)
    

