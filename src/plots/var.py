import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from pyvis.network import Network

plt.rcParams['font.family'] = 'serif'
main_color = '#0097a7'


def plot_params(data, model, fig_path=Path('.')):
    params = model.get_params()
    
    f, ax = plt.subplots(1, 2, figsize=(6, 4), 
                         gridspec_kw={'wspace': 0, 'width_ratios': [1, 0.01]})

    cmap = sns.color_palette("vlag", as_cmap=True)
    mx_val = np.max(np.abs(params-np.diag(np.diag(params))))
    norm = plt.Normalize(vmin=-mx_val, vmax=mx_val)
    plot = ax[0].matshow(params, cmap=cmap, norm=norm)

    labs = data.get_component_label()
    ax[0].set_xticks(np.arange(params.shape[0]))
    ax[0].set_yticks(np.arange(params.shape[0]))
    ax[0].set_xticklabels(labs)
    ax[0].set_yticklabels(labs)
    ax[0].tick_params(axis='x', top=False, labeltop=False, bottom=False, labelbottom=True)
    ax[0].tick_params(axis='y', left=False)

    f.colorbar(plot, cax=ax[1])

    if fig_path.is_dir():
        fig_path = fig_path/'VAR_params.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_corr(data, fig_path=Path('.')):
    residuals = data.time_courses()
    residuals = residuals.reshape((residuals.shape[0], -1))
    cor = np.corrcoef(residuals)

    f, ax = plt.subplots(1, 2, figsize=(6, 4), 
                         gridspec_kw={'wspace': 0, 'width_ratios': [1, 0.01]})
    cmap = sns.color_palette("vlag", as_cmap=True)
    mx_val = np.max(np.abs(cor-np.eye(cor.shape[0])))
    norm = plt.Normalize(vmin=-mx_val, vmax=mx_val)
    plot = ax[0].matshow(cor, cmap=cmap, norm=norm)

    labs = data.get_component_label()
    ax[0].set_xticks(np.arange(cor.shape[0]))
    ax[0].set_yticks(np.arange(cor.shape[0]))
    ax[0].set_xticklabels(labs)
    ax[0].set_yticklabels(labs)
    ax[0].tick_params(axis='x', top=False, labeltop=False, bottom=False, labelbottom=True)
    ax[0].tick_params(axis='y', left=False)

    f.colorbar(plot, cax=ax[1])

    if fig_path.is_dir():
        fig_path = fig_path/'VAR_cor.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_corr(data, model, fig_path=Path('.')):
    residuals = model.get_residuals()
    cor = np.corrcoef(residuals)

    f, ax = plt.subplots(1, 2, figsize=(6, 4), 
                         gridspec_kw={'wspace': 0, 'width_ratios': [1, 0.01]})
    cmap = sns.color_palette("vlag", as_cmap=True)
    mx_val = np.max(np.abs(cor-np.eye(cor.shape[0])))
    norm = plt.Normalize(vmin=-mx_val, vmax=mx_val)
    plot = ax[0].matshow(cor, cmap=cmap, norm=norm)

    labs = data.get_component_label()
    ax[0].set_xticks(np.arange(cor.shape[0]))
    ax[0].set_yticks(np.arange(cor.shape[0]))
    ax[0].set_xticklabels(labs)
    ax[0].set_yticklabels(labs)
    ax[0].tick_params(axis='x', top=False, labeltop=False, bottom=False, labelbottom=True)
    ax[0].tick_params(axis='y', left=False)

    f.colorbar(plot, cax=ax[1])

    if fig_path.is_dir():
        fig_path = fig_path/'VAR_res_cor.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_node_pos(comps, mask):
    ''' Gives components (x, y) coordinates based on its center of mass in the x=z_{max} plane.
    Args:
        comps: (np.ndarray) array of shape (n_comps, x, y, z)
    Returns:
        np.ndarray: array of shape (n_comps, 2) with the centroids coordinates.
    '''
    n_comps, nx, ny, nz = comps.shape

    x, y, z = np.meshgrid(np.arange(nx)/nx, np.arange(ny)/ny, np.arange(nz)/nz)
    coords = np.stack([x, y, z], axis=-1)[mask]
    cents = []
    for comp in range(n_comps):
        vals = comps[comp][mask]
        centroid = np.sum(coords*vals[:, None], axis=0)/vals.sum()
        cents.append(centroid[1:])
    cents = np.stack(cents, axis=0)
    return cents



def create_graph(adj_mat, labels, positions, thresh=0.01):
    n_nodes = adj_mat.shape[0]

    n_pix = 700
    positions[:, 1] = 1.0-positions[:, 1]/positions[:, 1].max()
    positions = positions*n_pix
    
    nt = Network(f'{n_pix}px', directed=True, filter_menu=False)
    nt.show_buttons(filter_=['edges','physics'])
    colors = iter(['green','yellow','orange','violet','maroon','grey','black','pink','cyan'])
    c = next(colors)
    lev = labels[0]

    # add nodes (colored by level)
    for i in range(n_nodes):
        # Change color
        if i>0:
            levn = labels[i]
            if levn != lev:
                c = next(colors)
            lev = levn
        pos = positions[i]
        nt.add_node(i, label=labels[i], color=c,physics=False, x=float(pos[0]),y=float(pos[1]))
        nt.nodes[i]['group'] = labels[i]

    # add edges according to threshold
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i==j:
                continue
            if np.abs(adj_mat[i,j]) < thresh:
                continue
            w = adj_mat[i,j]
            c = 'red' if w>0 else 'blue'
            nt.add_edge(i, j,weight=w,title=w,color=c,value=np.abs(w))
    
    return nt


def plot_graph(data, model, threshold=0.01, fpath=Path('.')):
    ''' Plots the graph of the VAR model.
    Args:
        data (GroupData): GroupData object with the data.
        model (VAR): VAR model.
        threshold (float, optional): Threshold to apply to the adjacency matrix. Defaults to None.
        fig_path (Path, optional): Path to save the figure. Defaults to current directory.
    '''
    adj_mat = model.get_params()
    adj_mat = adj_mat - np.diag(np.diag(adj_mat))
    pos = get_node_pos(data.get_components().transpose((3, 0, 1, 2)), data.get_mask())
    labs = data.get_component_label()
    nt = create_graph(adj_mat, labs, pos, thresh=threshold)
    nt.save_graph(str(fpath/'VAR_graph.html'))
    # G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    # nx.draw_networkx(G, pos=pos, labels={i: labs[i] for i in range(len(labs))}, arrows=True, 
    #                  )#edge_color=)
    # plt.savefig(fpath/'VAR_graph.png', dpi=300, bbox_inches='tight')