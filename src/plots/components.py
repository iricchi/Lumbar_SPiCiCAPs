import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path
from tqdm.auto import tqdm

from ..data.paths import PAM50_CERVICAL_PATH, PAM50_LUMBAR_PATH, SPINAL_LEVELS
plt.rcParams['font.family'] = 'serif'
main_color = '#0097a7'


class ComponentViewer:
    def __init__(self, data, template_path=PAM50_LUMBAR_PATH):
        self.data = data
        self.template = None
        if template_path is not None:
            self.template = nib.load(template_path).get_fdata()
        # else:
        #     self.template = self.data.get_epi().mean(axis=-1)*self.data.mask

    def change_data(self, data):
        self.data = data

    def plot(self, component, coords=None, alpha=0.7, 
                   show_level=True, mask=False):
        ''' Plot component overlayed on top of the template.
        Args:
            component (int): Index of the component to plot.
            coords (tuple, optional): Tuple of coordinates (x, y, z) to plot. Defaults to None.
            alpha (float, optional): Alpha value for the overlay. Defaults to 0.7.
        '''
        # Get corresponding component
        max_val = self.data.get_components().max()
        vol = np.abs(self.data.get_components()[:, :, :, component])
        f, axs = plt.subplots(1, 4, figsize=(12, 4), 
                              gridspec_kw={'width_ratios': [1.0, 1.0, 1.0, 0.05], 
                                           'wspace': 0.1, 'hspace': 0.05})
        if coords is None:
            if len(np.unique(vol))==2:
                x = np.argmax(vol.sum(axis=(1, 2)))
                y = np.argmax(vol.sum(axis=(0, 2)))
                z = np.argmax(vol.sum(axis=(0, 1)))
            else:
                x, y, z = np.unravel_index(np.argmax(vol), vol.shape)
        else:
            x, y, z = coords
        
        axs[0].set_title(f'X={x}', color='white')
        axs[1].set_title(f'Y={y}', color='white')
        axs[2].set_title(f'Z={z}', color='white')
        if self.template is not None:
            # Plot template
            axs[0].imshow(self.template[x, :, ::-1].T, cmap='gray')
            axs[1].imshow(self.template[:, y, ::-1].T, cmap='gray')
            axs[2].imshow(self.template[:, ::-1, z].T, cmap='gray')
            

        label, score = self.data.get_component_label(component, True)
        alpha2 = 0.5
        if show_level:
            spinal_level = self.data.template
            level_idx = np.where(SPINAL_LEVELS==label)[0][0]
            level_mask = (spinal_level==level_idx).astype(int)

            cmap = sns.blend_palette(['#000000', 'blue'], n_colors=2, as_cmap=True)
            axs[0].imshow(level_mask[x, :, ::-1].T, cmap=cmap, alpha=alpha2, norm=plt.Normalize(0, 1))
            axs[1].imshow(level_mask[:, y, ::-1].T, cmap=cmap, alpha=alpha2, norm=plt.Normalize(0, 1))
            axs[2].imshow(level_mask[:, ::-1, z].T, cmap=cmap, alpha=alpha2, norm=plt.Normalize(0, 1))

        if mask:
            mask = self.data.mask
            axs[0].imshow(mask[x, :, ::-1].T, cmap='gray', alpha=alpha2, norm=plt.Normalize(0, 1))
            axs[1].imshow(mask[:, y, ::-1].T, cmap='gray', alpha=alpha2, norm=plt.Normalize(0, 1))
            axs[2].imshow(mask[:, ::-1, z].T, cmap='gray', alpha=alpha2, norm=plt.Normalize(0, 1))

        norm = plt.Normalize(max_val*0.15, max_val)
        # Plot component overlayed on top
        x_view = axs[0].imshow(vol[x, :, ::-1].T, cmap='hot', alpha=alpha, norm=norm)
        y_view = axs[1].imshow(vol[:, y, ::-1].T, cmap='hot', alpha=alpha, norm=norm)
        z_view = axs[2].imshow(vol[:, ::-1, z].T, cmap='hot', alpha=alpha, norm=norm)

        # Remove all ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        f.colorbar(x_view, cax=axs[3])
        axs[3].set_ylabel('Z-score', color='white')
        axs[3].yaxis.set_label_position('left')
        # Set xticks and color to white
        axs[3].tick_params(axis='x', colors='white')
        axs[3].tick_params(axis='y', colors='white')
        # Set background color of the figure to black
        f.patch.set_facecolor('black')
        if score is not None:
            plt.suptitle(f'Component {component}: {label}, (DICE={score:.2f})', color='white')
        else:
            plt.suptitle(f'Component {component}', color='white')
        return f
    
    def plot_all(self, fig_path, **kwargs):
        for idx in range(self.data.n_components):
            f = self.plot(idx, **kwargs)
            plt.savefig(fig_path/f'component_{idx}.png', bbox_inches='tight', dpi=300)
            plt.close()


def plot_consensus(data, ks, fpath=Path('.')):
    '''
    Plots the consensus matrix for different values of k.
    Args:
        data (Data): Data object.
        ks (list): List of values of k to plot.
        fpath (Path, optional): Path to save the figures.
    '''
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    meds = []
    for k in ks:
        data.change_components(k)
        ax.boxplot(data.icaps_result.consensus, positions=[k], widths=0.5, showfliers=False, 
                   medianprops={'color': main_color})
        meds.append(np.median(data.icaps_result.consensus))

    ax.plot(ks, meds, color=main_color, linewidth=2)
    ax.set_xlabel('Components')
    ax.set_ylabel('Consensus score')
    ax.set_ylim(0, 1)

    xtcks = np.arange(0, np.max(ks)+5, 5)[1:]
    ax.set_xticks(xtcks)
    ax.set_xticklabels(xtcks)
    # Remove splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add grid in both axes
    ax.grid(visible=True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, axis='both')

    plt.savefig(fpath/'consensus.png', dpi=300)
    plt.close()
        

def plot_component_comparison(sim_matrix, level_labels=None, fig_path=Path('.')):
    ''' Plots the similarity matrix between iCAPs and spinal cord levels.
    Args:
        sim_matrix (np.ndarray): Matrix of shape (n_components, n_levels) with the similarity scores.
        level_labels (np.ndarray, optional): Array of shape (n_levels,) with the labels of the spinal cord levels.
        fig_path (Path, optional): Path to save the figure. Defaults to current directory.
    '''
    f, axs = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [1.0, 0.03], 'wspace': 0.01})

    cmap = sns.blend_palette(['white', main_color], as_cmap=True)
    mx_val = np.nanmax(sim_matrix[::-1])
    mat_plot = axs[0].matshow(sim_matrix[::-1], cmap=cmap, 
                              aspect='auto', norm=plt.Normalize(0, mx_val))
    axs[0].set_xlabel('Spinal cord level')
    axs[0].set_ylabel('iCAPs', labelpad=-5)
    if level_labels is not None:
        axs[0].set_xticks(np.arange(len(level_labels)))
        axs[0].set_xticklabels(level_labels)
        axs[0].set_xlabel('Spinal cord level')
        axs[0].xaxis.set_label_position('bottom')
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labeltop=False, labelbottom=True)
    # axs[0].spines['top'].set_visible(False)
    # axs[0].spines['right'].set_visible(False)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True, labelright=False)
    axs[0].set_yticks([-0.5, sim_matrix.shape[0]-0.5])
    axs[0].set_yticklabels([sim_matrix.shape[0], 1])
    # Colorbar stuff
    f.colorbar(mat_plot, cax=axs[1])
    axs[1].set_ylabel('Dice score', labelpad=-5)
    #axs[1].yaxis.set_label_position('left')
    
    axs[1].set_yticks([0, mx_val])
    axs[1].set_yticklabels([0, f'{mx_val:.1f}'])
    axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=True)

    # Check if fig_path includes file extension
    if fig_path is None:
        return f, axs
    if not fig_path.suffix:
        plt.savefig(fig_path/f'level_similarity.png', dpi=300)
    else:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_component_comparison_subject(sims, labels, fig_path=Path('.')):
    f, axs = plt.subplots(figsize=(16, 5), ncols=1, nrows=2, sharex=True, 
                     gridspec_kw={'hspace': 0, 'height_ratios': [1, 0.05]})
    ax1 = axs[0]
    ax2 = axs[1]
    idx = 1
    deltax = 1.0
    for i in range(sims.shape[0]):
        start_idx = idx
        clab = labels[0]
        lab_start = start_idx
        for j in range(sims.shape[1]):
            if (j>0) and (labels[j]!=clab):
                ax2.plot([lab_start-0.3, idx-0.7], [0.15, 0.15], c='k')
                ax2.text((lab_start+idx-1)/2, 0.13, clab, ha='center', va='center', fontsize=6)
                lab_start = idx
                clab = labels[j]
            coups = sims[i, j]
            ax1.boxplot(coups[~np.isnan(coups)], positions=[idx], widths=0.3, labels=[labels[j]])
            idx += deltax
        # Last label
        ax2.plot([lab_start-0.3, idx-0.7], [0.15, 0.15], c='k')
        ax2.text((lab_start+idx-1)/2, 0.13, clab, ha='center', va='center', fontsize=6)

        idx += deltax
        ax2.plot([start_idx, idx-2], [0.1, 0.1], c='k')
        ax2.text((start_idx+idx-2)/2, 0.05, labels[i], ha='center', va='center')

    ax1.set_ylabel('Dice score')
    ax1.set_xticks([])
    # Remove all axes lines from ax2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    # Set x ticks in ax1 to visible
    ax1.tick_params(axis='x', which='major', labelsize=8, bottom=True, labelbottom=True)
    ax2.tick_params(axis='x', which='major', labelsize=8, bottom=False, labelbottom=False)
    ax1.grid(axis='y', visible=True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    if fig_path is None:
        return f, axs
    if fig_path.is_dir():
        fig_path = fig_path/'positive_per_subject.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
