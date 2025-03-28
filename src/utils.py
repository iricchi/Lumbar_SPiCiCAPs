import numpy as np
import networkx as nx


def match(sim):
    ''' Computes maximum weight matching between two sets of nodes.
    Args:
        sim (np.ndarray): similarity matrix of shape (n_nodes1, n_nodes2).
    Returns:
        np.ndarray: array of shape (n_nodes1,) containing the index of the best match for each node.
    '''
    N = sim.shape[0]
    M = sim.shape[1]
    T = N+M
    # Create adjacency matrix of the whole graph
    adj = np.zeros((N+M, N+M))
    adj[:N, N:T] = sim
    adj[N:T, :N] = sim.T
    graph = nx.from_numpy_array(adj)
    # Find optimal assignment
    matches = nx.algorithms.max_weight_matching(graph)
    # Retrieve values
    best_match = np.zeros(N, dtype=int)
    for m1, m2 in matches:
        if m1 > m2:
            m1, m2 = m2, m1
        best_match[m1] = m2-N
    return best_match


def jaccard_similarity(signal1, signal2):
    ''' Computes the Jaccard similarity between two signals.
    Args:
        signal1: array of shape (n_samples,) or (n_reps, n_samples)
        signal2: array of shape (n_samples,) or (n_reps, n_samples)
    Returns:
        float: Jaccard similarity between the two signals.
    '''
    assert signal1.shape==signal2.shape
    if len(signal1.shape)==1:
        non_zero = (signal1!=0) | (signal2!=0)
        intersec = np.sum(signal1[non_zero]==signal2[non_zero])
        union = np.sum(non_zero)
        if union==0:
            return 0
        return intersec/union

    non_zero = (signal1!=0) | (signal2!=0)
    are_equal = np.zeros((signal1.shape))
    are_equal = np.equal(signal1, signal2, where=non_zero, out=are_equal)
    intersec = np.sum(are_equal, axis=1)
    union = np.sum(non_zero, axis=1)
    iou = np.zeros((signal1.shape[0]))
    iou[union!=0] = intersec[union!=0]/union[union!=0]
    return iou


def compute_couplings(timecourses, anti=False, per_subject=False):
    ''' Computes all pair-wise jaccard similarities between time courses.
    Args:
        timecourses (np.ndarray): array of shape (n_components, n_subjects, n_times) or (n_components, n_samples).
                                  arrays can have the following values: {-1, 0, 1}
        anti (bool): whether to compute the anti-correlation.
    '''
    n_components = timecourses.shape[0]
    if len(timecourses.shape)==2:
        couplings = np.zeros((n_components, n_components))
    else: 
        n_subjects = timecourses.shape[1]
        n_times = timecourses.shape[2]
        if not per_subject:
            timecourses = timecourses.reshape((n_components, n_subjects*n_times))
            couplings = np.zeros((n_components, n_components))
        else:
            couplings = np.zeros((n_components, n_components, n_subjects))

    # Compute pair-wise couplings
    for i in range(n_components):
        for j in range(n_components):
            if i>j:
                couplings[i][j] = couplings[j][i]
            else:
                if anti:
                    couplings[i][j] = jaccard_similarity(timecourses[i], -timecourses[j])
                else:
                    couplings[i][j] = jaccard_similarity(timecourses[i], timecourses[j])
    return couplings