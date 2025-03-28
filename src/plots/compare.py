import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

main_color = '#0097a7'


def plot_sim_mat(comparer, fig_path=None):
    sims = comparer.get_sims()
    folds = comparer.get_folds()
    nfolds = comparer.n_folds()

    fchange = np.append(np.diff(folds), 1)!=0
    fchange = np.insert(np.where(fchange)[0], 0, 0)
    f, ax = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [1, 0.05], 'wspace': 0.01})
    cmap = sns.blend_palette(['white', main_color], as_cmap=True)
    ax[0].matshow(sims, cmap=cmap, vmin=0, vmax=1)
    # Add black lines in changes
    for i in fchange:
        if i==0:
            pos = i-0.5
        else:
            pos = i+0.5
        ax[0].axvline(pos, color='black')
        ax[0].axhline(pos, color='black')
    # Add ticklabels in the middle
    pos = (fchange[:-1]+fchange[1:])/2
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels([f'Fold {i+1}' for i in range(nfolds)])
    ax[0].set_yticks(pos)
    ax[0].set_yticklabels([f'Fold {i+1}' for i in range(nfolds)])
    ax[0].tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=True)
    ax[0].tick_params(axis='y', left=False, right=False, labelleft=True, labelright=False)
    
    f.colorbar(ax[0].images[0], cax=ax[1])
    ax[1].set_ylabel('Dice similarity')
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels([0, 1])
    if fig_path is None:
        return f, ax

    if fig_path.is_dir():
        fig_path = fig_path/'sim_mat.pdf'
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_single_stability(stability, fig_path):
    ax = sns.boxplot(y=stability)
    ax.set_ylabel('Stability')
    ax = sns.swarmplot(y=stability, color=main_color)
    if fig_path.is_dir():
        fig_path = fig_path/'stability.pdf'
    plt.savefig(fig_path)
    plt.close()


def plot_all_stabilities(comps, stabs, fig_path):
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.boxplot(stabs, labels=comps, positions=comps, medianprops={'color': main_color})
    meds = [np.median(s) for s in stabs]
    ax.plot(comps, meds, color=main_color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Dice score')
    ax.grid(visible=True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, axis='both')
    ax.set_ylim(0, 1)
    xtcks = np.arange(0, np.max(comps)+5, 5)[1:]
    ax.set_xticks(xtcks)
    ax.set_xticklabels(xtcks)
    if fig_path.is_dir():
        fig_path = fig_path/'all_stabilities.pdf'
    plt.savefig(fig_path, dpi=300)
    plt.close()
