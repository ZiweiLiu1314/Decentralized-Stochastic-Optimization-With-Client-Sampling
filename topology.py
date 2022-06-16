import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx


def create_two_rings(n_cores):
    assert (n_cores % 2) == 0
    assert n_cores >= 3
    # create W1 first
    W = np.zeros(shape=(n_cores, n_cores))
    value = 1./3 if n_cores >= 3 else 1./2
    np.fill_diagonal(W, value)
    np.fill_diagonal(W[1:], value, wrap=False)
    np.fill_diagonal(W[:, 1:], value, wrap=False)
    W[0, n_cores - 1] = value
    W[n_cores - 1, 0] = value
    
    W1 = W
    
    group_1 = np.arange(0, n_cores // 2)
    group_2 = np.arange( n_cores // 2, n_cores)
    # print(group_1, group_2)
    W = np.zeros(shape=(n_cores, n_cores))
    value = 1./3 if n_cores >= 3 else 1./2
    np.fill_diagonal(W, value)
    for i in range(0, n_cores // 2):
        W[group_1[i], group_2[n_cores // 2 - i - 1]] = value
        W[group_2[n_cores // 2 - i - 1], group_1[i]] = value
        if i > 0:
            W[group_1[i], group_2[n_cores // 2 - i]] = value
            W[group_2[n_cores // 2 - i], group_1[i]] = value
    W[group_1[0]][group_2[0]] = value
    W[group_2[0]][group_1[0]] = value
    W2 = W
    return W1, W2

def create_mixing_matrix(topology, n_cores):
    assert topology in ['ring', 'centralized', 'grid', 'avg_two_ring']
    if topology == 'avg_two_ring':
        W1, W2 = create_two_rings(n_cores)
        return (W1 + W2) * 0.5
    elif topology == 'ring':
        W = np.zeros(shape=(n_cores, n_cores))
        value = 1./3 if n_cores >= 3 else 1./2
        np.fill_diagonal(W, value)
        np.fill_diagonal(W[1:], value, wrap=False)
        np.fill_diagonal(W[:, 1:], value, wrap=False)
        W[0, n_cores - 1] = value
        W[n_cores - 1, 0] = value
        return W
    elif topology == 'centralized':
        W = np.ones((n_cores, n_cores), dtype=np.float64) / n_cores
        return W
    else:
        assert int(np.sqrt(n_cores)) ** 2 == n_cores
        G = networkx.generators.lattice.grid_2d_graph(int(np.sqrt(n_cores)),
                                            int(np.sqrt(n_cores)), periodic=True)
        W = networkx.adjacency_matrix(G).toarray()
        for i in range(0, W.shape[0]):
            W[i][i] = 1
        W = W/5
        return W



class MixingMatrix:
    def __call__(self, current_iter):
        pass

class FixedMixingMatrix(MixingMatrix):
    def __init__(self, topology_name, n_cores):
        self.W = create_mixing_matrix(topology_name, n_cores)
    def __call__(self, current_iter):
        return self.W
