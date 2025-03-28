
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from shutil import rmtree

from ..utils import compute_couplings

main_color = '#0097a7'


def plot_coactivation(data, fig_path):
    fig_path = fig_path/'couplings'
    fig_path.mkdir(exist_ok=True)

    #timecourses = data.time_courses(binary=True, signed=True)
    res = data.get_results()
    # Couplings
    couplings = res.get_couplings()#compute_couplings(timecourses)
    anti_couplings = res.get_couplings(anti=True)#compute_couplings(timecourses, anti=True)

    # Plot couplings
    cmap = sns.blend_palette(['white', main_color], as_cmap=True)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    mx = np.nanmax(couplings)
    norm = plt.Normalize(0, mx)
    cop_plot = ax.matshow(couplings, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(couplings.shape[0]))
    ax.set_yticks(np.arange(couplings.shape[0]))
    ax.set_xticklabels(data.get_component_label())
    ax.tick_params(axis='x', top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.set_yticklabels(data.get_component_label())
    fig.colorbar(cop_plot, label='Jaccard sim')
    plt.savefig(fig_path/'positive.png')
    plt.close()

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    mx = np.nanmax(anti_couplings)
    norm = plt.Normalize(0, mx)
    cop_plot = ax.matshow(anti_couplings, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(couplings.shape[0]))
    ax.set_yticks(np.arange(couplings.shape[0]))
    ax.set_xticklabels(data.get_component_label())
    ax.tick_params(axis='x', top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.set_yticklabels(data.get_component_label())
    fig.colorbar(cop_plot, label='Jaccard sim')
    plt.savefig(fig_path/'negative.png')
    plt.close()


def plot_coactivation_per_subject(data, fig_path):
    fig_path = fig_path/'couplings'
    fig_path.mkdir(exist_ok=True)

    #timecourses = data.time_courses(binary=True, signed=True)
    res = data.get_results()
    couplings = res.get_couplings(per_subject=True)#compute_couplings(timecourses, per_subject=True)
    anti_couplings = res.get_couplings(per_subject=True, anti=True)#compute_couplings(timecourses, anti=True, per_subject=True)
    f, axs = plt.subplots(figsize=(10, 5), ncols=1, nrows=2, sharex=True, 
                     gridspec_kw={'hspace': 0.1, 'height_ratios': [1, 0.05]})
    ax1 = axs[0]
    ax2 = axs[1]
    idx = 1
    deltax = 1.0
    for i in range(couplings.shape[0]):
        start_idx = idx
        for j in range(couplings.shape[1]):
            if i==j:
                continue
            coups = couplings[i, j]
            ax1.boxplot(coups[~np.isnan(coups)], positions=[idx], widths=0.3, labels=[data.get_component_label()[j]])
            idx += deltax
        idx += deltax
        ax2.plot([start_idx, idx-2], [0.1, 0.1], c='k')
        ax2.text((start_idx+idx-2)/2, 0.09, data.get_component_label()[i], ha='center', va='center')

    ax1.set_ylabel('Jaccard similarity')
    # Remove all axes lines from ax2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    # Set x ticks in ax1 to visible
    ax1.tick_params(axis='x', which='major', labelsize=8, bottom=True, labelbottom=True)
    ax2.tick_params(axis='x', which='major', labelsize=8, bottom=False, labelbottom=False)
    #ax2.set_xticks([])
    ax1.set_title('Couplings')
    plt.savefig(fig_path/'positive_per_subject.png')
    plt.close()

    f, axs = plt.subplots(figsize=(10, 5), ncols=1, nrows=2, sharex=True, 
                     gridspec_kw={'hspace': 0.1, 'height_ratios': [1, 0.05]})
    ax1 = axs[0]
    ax2 = axs[1]
    idx = 1
    for i in range(anti_couplings.shape[0]):
        start_idx = idx
        for j in range(anti_couplings.shape[1]):
            if i==j:
                continue
            coups = anti_couplings[i, j]
            ax1.boxplot(coups[~np.isnan(coups)], positions=[idx], widths=0.5, labels=[data.get_component_label()[j]])
            idx += 1
        idx += 1
        ax2.plot([start_idx, idx-2], [0.1, 0.1], c='k')
        ax2.text((start_idx+idx-2)/2, 0.09, data.get_component_label()[i], ha='center', va='center')
    ax1.set_ylabel('Jaccard similarity')
    # Remove all axes lines from ax2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    # Set x ticks in ax1 to visible
    ax1.tick_params(axis='x', which='major', labelsize=8, bottom=True, labelbottom=True)
    ax2.tick_params(axis='x', which='major', labelsize=8, bottom=False, labelbottom=False)
    #ax2.set_xticks([])
    ax1.set_title('Anti-couplings')
    plt.savefig(fig_path/'negative_per_subject.png')
    plt.close()


def plot_timecourses(data, fpath, binary=False):
    if binary:
        fpath = fpath/'timecourses_binary'
    else:
        fpath = fpath/'timecourses'
    if fpath.exists():
        rmtree(fpath)
    fpath.mkdir()

    timecourses = data.time_courses(binary=binary)
    for i, timecourse in enumerate(timecourses):
        f, axs = plt.subplots(timecourse.shape[0]//2+1, 2, figsize=(6, 12), sharex=True, sharey=True,
                              gridspec_kw={'hspace': 0.6, 'wspace': 0.1})
        lab, score = data.get_component_label(i, score=True)
        plt.suptitle(f'Component {i}: {lab}, {score:.2f}')
        axs = axs.flatten()
        for j, tc in enumerate(timecourse):
            axs[j].plot(tc)
            axs[j].set_title(f'{data.get_results().subject(j)}')
        plt.savefig(fpath/f'timecourse_{i}.png')
        plt.close()


def plot_correlation(data, fig_path):
    tcs = data.time_courses()
    tcs_flat = tcs.reshape((tcs.shape[0], -1))
    
    cor = np.corrcoef(tcs_flat)
    vmax = np.max(np.abs(cor-np.eye(cor.shape[0])))
    norm = plt.Normalize(-vmax, vmax)

    cmap = sns.color_palette("vlag", as_cmap=True)
    plt.matshow(cor, norm=norm, cmap=cmap)
    plt.savefig(fig_path/'correlation.png')