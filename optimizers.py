import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx
from utils import consensus_distance

def optimize_decentralized(X, topology, A, B, gamma, sigma, num_iter=100):
    # X.shape = (num_dim, num_nodes)
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)
    
    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)
    errors = [consensus_distance(X_iter, A, B)]
    for i in range(0, num_iter):
        W = topology(i)
        AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B) # shape (num_nodes, num_dim)
        grad = np.einsum("ijk,ij->ik", A, AXmB) # shape (num_nodes, num_dim)
        noise = np.random.normal(0, np.sqrt(sigma / num_dim), size=X.shape)
        X_iter = X_iter - gamma * (grad.T + noise)
        X_iter = X_iter.dot(W)
        errors += [consensus_distance(X_iter, A, B)]
    return errors, X_iter
