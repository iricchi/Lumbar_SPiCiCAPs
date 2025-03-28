import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

plt.rcParams['font.family'] = 'serif'
main_color = '#0097a7'


def plot_merger_iou(merger, boxplot=False, 
                    fpath=Path('.')):
    
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    if boxplot:
        out_path = fpath/'iou_boxplot.png'
        ax.boxplot(merger.iou, medianprops={'color': main_color}, positions=np.arange(len(merger.iou)))
        ax.set_ylabel('IOU')
    else:
        ious = [vals.mean() for vals in merger.iou]
        out_path = fpath/'iou.png'
        ax.plot(ious, color=main_color, linewidth=2)
        ax.set_ylabel('Average IOU')
    ax.set_xlabel('Merge step')
    
    # Remove splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tree(merger, data=None, fpath=Path('.')):
    if fpath.is_dir():
        fpath = fpath/'tree.png'
    f, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-0.5, merger.n_components+0.5)
    ax.axis('off')
    # Sort leaves according to final cluster
    if data is not None:
        ord = data.icaps_result._old2new_order
        if len(merger.component_childs) == 1:
            leaves = merger.component_childs[0]
        else:
            leaves = np.concatenate([merger.component_childs[i] for i in ord])
    else:
        leaves = np.concatenate(np.array(merger.component_childs)[ord])
    x = np.arange(len(leaves))
    y = 0
    node_pos = np.ones((merger.n_components, 2), dtype=float)
    for x_pos, name in zip(x, leaves):
        lname = merger.old_labels[name]
        lscore = merger.old_sims[name]
        ax.text(x_pos, y-0.2, lname, ha='center', va='center', fontsize=8)
        ax.text(x_pos, y-0.5, f'{lscore:.2f}', ha='center', va='center', fontsize=8)
        node_pos[name] = [x_pos, y]

    # Iterate through merges
    max_y = 0
    for merge in merger.merges:
        i, j, _ = merge
        x_i, y_i = node_pos[i]
        x_j, y_j = node_pos[j]
        x_new = (x_i + x_j)/2
        y_new = max(y_i, y_j) + 1
        node_pos[i, 0] = x_new
        node_pos[i, 1] = y_new
        node_pos = np.delete(node_pos, j, axis=0)
        if y_new > max_y:
            max_y = y_new
        # Draw lines
        # Vertical
        ax.vlines(x_i, y_i, y_new, color='black')
        ax.vlines(x_j, y_j, y_new, color='black')
        # Horizontal
        ax.hlines(y_new, x_i, x_j, color='black')

    # Draw final lines
    # Sort by x position
    node_pos = node_pos[np.argsort(node_pos[:, 0])]
    for i, (x, y) in enumerate(node_pos):
        ax.vlines(x, y, max_y+1, color='black')
        name, score = data.icaps_result.get_component_label(i, score=True)
        ax.text(x, max_y+1.4, name, ha='center', va='center', fontsize=8)
        ax.text(x, max_y+1.2, f'{score:.2f}', ha='center', va='center', fontsize=8)
    plt.savefig(fpath, dpi=300)
    plt.close()