import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx


def generate_functions(num_nodes, num_dim, zeta):
    A = [1 / np.sqrt(num_nodes) * np.eye(num_dim) * (i + 1) for i in range(0, num_nodes)]
    B = [np.random.normal(0, np.sqrt(zeta) / (i + 1), size=num_dim) for i in range(0, num_nodes)]
    # B = [np.ones(num_dim) * np.sqrt(zeta) / (i + 1) for i in range(0, num_nodes)]
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)
    return np.array(A), np.array(B)

def generate_consensus_functions(num_nodes, num_dim):
    # f_i = || A_i (x - b_i) ||_2^2
    A = [np.identity(num_dim) for i in range(0, num_nodes)]
    B = [1 + i + np.random.normal(size=num_dim) for i in range(0, num_nodes)]
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)
    return np.array(A), np.array(B)

def consensus_distance(X, A, B): # ||x-x*||^2
    # X.shape = (num_dim, num_nodes)
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)

    x_star = np.linalg.inv(np.einsum("ikj,ikl->jl", A, A)).dot(np.einsum("ijk,ij->k", A, B))
    
    num_nodes = X.shape[1]
    dist = [np.linalg.norm(X[:,i] - x_star) ** 2 for i in range(0, num_nodes)]
    return np.mean(dist)

