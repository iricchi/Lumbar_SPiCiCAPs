import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path
from shutil import rmtree
from scipy.signal import periodogram, correlate

sys.path.append('.')
from src.data import GroupData
import src.data.paths as paths
import src.plots.time_courses as tc_plots

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 9})


def find_plateau(power, threshold=0):
    ''' Find the first index where the power drops below the threshold.
    Args:
        power (np.ndarray): array of shape (n_freqs,)
        threshold (float): threshold to use
    Returns:
        int: index of the first frequency where the power drops below the threshold
    '''
    is_below = power<threshold
    if is_below.sum()==0:
        return power.shape[0]
    last_idx = np.where(np.diff(is_below)==1)[0][-1]
    return last_idx


def plot_psd(data, fpath):
    fpath = fpath/'psd.png'

    ncols = np.ceil((data.n_components)/2).astype(int)
    f, axs = plt.subplots(2, ncols, figsize=(10, 10), sharex=True, sharey=True, 
                          gridspec_kw={'hspace': 0.1, 'wspace': 0})
    axs = axs.flatten()
    timecourses = data.time_courses()
    for i, timecourse in enumerate(timecourses):
        freqs, psd = periodogram(timecourse, fs=data.get_results().fs, axis=1)
        mean_power = np.mean(psd, axis=0)
        # Cut freqs where power is too low
        cut = find_plateau(mean_power)
        freqs = freqs[:cut]
        psd = psd[:, :cut]
        mean_power = np.mean(psd, axis=0)
        std_power = np.std(psd, axis=0)
        axs[i].plot(freqs, mean_power)
        up_band = mean_power+1.96*std_power/np.sqrt(psd.shape[0])
        low_band = mean_power-1.96*std_power/np.sqrt(psd.shape[0])
        low_band[low_band<0] = 0
        axs[i].fill_between(freqs, low_band, up_band, alpha=0.5)
        axs[i].set_title(f'Component {i+1}: {data.get_component_label()[i]}, {data.label_scores[i]:.2f}')
        if i%2==0:
            axs[i].set_ylabel('Power')

        if i>=ncols:
            axs[i].set_xlabel('Frequency (Hz)')
    
    plt.savefig(fpath)
    plt.close()


def plot_correlations(data, fpath):
    fpath = fpath/'correlation.png'
    f, axs = plt.subplots(data.n_components, data.n_components, 
                          figsize=(10, 10), sharex=True, sharey=True, 
                          gridspec_kw={'hspace': 0.1, 'wspace': 0})
    
    # Create correlation matrix
    tcs = data.time_courses()
    for i, timecourse in enumerate(tcs):
        for j, timecourse2 in enumerate(tcs):
            if i > j:
                continue
            cor = correlate(timecourse.T, timecourse2.T)
            axs[i, j].plot(cor)
    plt.savefig(fpath)


def plot_raster(data, fig_path):
    fig_path = fig_path/'raster.png'

    timecourse = data.time_courses(binary=True)
    f, axs = plt.subplots(timecourse.shape[1]//2+1, 2, figsize=(10, 20), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0, 'wspace': 0.1})
    axs = axs.flatten()
    for sbj in range(timecourse.shape[1]):
        y_points, x_points = np.where(timecourse[:, sbj, :])
        axs[sbj].scatter(x_points, y_points, s=1, marker='|', c='k')
        if sbj%2==0:
            axs[sbj].set_ylabel('Component')
        if sbj>=timecourse.shape[1]-2:
            axs[sbj].set_xlabel('Time (s)')
    plt.savefig(fig_path)
    plt.close()



def main(args):
    if args.lumbar:
        path = paths.LUMBAR_PATH
        #path = Path('/media/miplab-nas2/Data2/SpinalCord/Spinal_fMRI_Lumbar/')/'iCAPs_results'/'PAM50_all_Alpha_5_950DOT05'
        fpath = Path(args.fpath)/'lumbar'
        template_path = paths.LUMBAR_TEMPLATE
    else:
        path = paths.CERVICAL_PATH
        fpath = Path(args.fpath)/'cervical'
        template_path = paths.CERVICAL_TEMPLATE
    fpath = fpath/f'K_{args.n_clusters}'/'time_courses'
    fpath.mkdir(parents=True, exist_ok=True)
    data = GroupData(path, n_components=args.n_clusters, 
                     template_path=template_path, verbose=args.verbose)
    data.compare_components()

    tc_plots.plot_coactivation(data, fpath)
    tc_plots.plot_coactivation_per_subject(data, fpath)
    tc_plots.plot_timecourses(data, fpath)
    tc_plots.plot_timecourses(data, fpath, True)
    tc_plots.plot_correlation(data, fpath)
    # plot_raster(data, fpath)
    # plot_psd(data, fpath)
    # plot_correlations(data, fpath)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='figures')
    parser.add_argument('-k', '--n_clusters', type=int, default=4)
    parser.add_argument('--lumbar', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
    