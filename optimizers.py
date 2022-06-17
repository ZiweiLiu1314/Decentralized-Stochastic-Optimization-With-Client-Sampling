import numpy as np
import matplotlib.pyplot as plt
from utils import consensus_distance, return_step_size, sampling, sampling_s, sampling_grid, shuffling, shuffling_back
from topology import FixedMixingMatrix

# probability of node failing 
def optimize_decentralized(X, A, B, zeta, sigma, sampling_method, sampling_param, shuffle, num_iter): 
    # getting dim & number of nodes 
    num_dim, num_nodes = X.shape
    topology_str = ["ring", "centralized", "grid"]
    errors = {}
    for curr_topology in topology_str:
      X_iter = np.copy(X)
      errors[curr_topology] = [consensus_distance(X_iter, A, B)]
      
      gamma = return_step_size((curr_topology, zeta, sigma))
      for i in range(0, num_iter):
          AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B) # shape (num_nodes, num_dim)
          grad = np.einsum("ijk,ij->ik", A, AXmB) # shape (num_nodes, num_dim)

          if sampling_method == "failure_prob":
            thr = sampling_param
            # sampled index, sub X, and grad
            if curr_topology is not "grid":        
              sampledIndex, X_curr, grad_curr = sampling(thr, num_nodes, X_iter, grad)
            else:
              sampledIndex, X_curr, grad_curr = sampling_grid(thr, num_nodes, X_iter, grad)

          elif sampling_method == "fixed_s":
            s = sampling_param
            assert int(np.sqrt(s)) ** 2 == s, "number of workers should be square of an int"
            sampledIndex, X_curr, grad_curr = sampling_s(s, num_nodes, X_iter, grad)
          numberOfSampled = np.sum(sampledIndex)
          # create W 
          if numberOfSampled != 0:
            topology = FixedMixingMatrix(curr_topology, numberOfSampled)
            W_curr = topology(i)
            noise = np.random.normal(0, np.sqrt(sigma / num_dim), size=X_curr.shape)
            if shuffle:
              # shuffling 
              index, X_curr_sh, grad_curr_sh = shuffling(X_curr, grad_curr)
              # update
              X_temp_sh = X_curr_sh - gamma * (grad_curr_sh.T + noise)
              X_next_sh = X_temp_sh.dot(W_curr)
              # shuffling_back 
              X_next = shuffling_back(index, X_next_sh)
            else:
              X_temp = X_curr - gamma * (grad_curr.T + noise)
              X_next = X_temp.dot(W_curr)
            X_iter[:, sampledIndex] = X_next
          errors[curr_topology] += [consensus_distance(X_iter, A, B)]
            # print('X_next:', X_next)
    return errors, X_iter

# probability of node failing 
def optimize_decentralized_one_topology(X, A, B, sigma, sampling_method, sampling_param, shuffle, num_iter, topology_str, gamma): 
    # getting dim & number of nodes 
    num_dim, num_nodes = X.shape
    errors = {}
    for curr_topology in topology_str:
      X_iter = np.copy(X)
      errors[curr_topology] = [consensus_distance(X_iter, A, B)]
      
      # gamma = return_step_size((curr_topology, zeta, sigma))
      for i in range(0, num_iter):
          AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B) # shape (num_nodes, num_dim)
          grad = np.einsum("ijk,ij->ik", A, AXmB) # shape (num_nodes, num_dim)

          if sampling_method == "failure_prob":
            thr = sampling_param
            # sampled index, sub X, and grad
            if curr_topology is not "grid":        
              sampledIndex, X_curr, grad_curr = sampling(thr, num_nodes, X_iter, grad)
            else:
              sampledIndex, X_curr, grad_curr = sampling_grid(thr, num_nodes, X_iter, grad)

          elif sampling_method == "fixed_s":
            s = sampling_param
            assert int(np.sqrt(s)) ** 2 == s, "number of workers should be square of an int"
            sampledIndex, X_curr, grad_curr = sampling_s(s, num_nodes, X_iter, grad)
          numberOfSampled = np.sum(sampledIndex)
          # create W 
          if numberOfSampled != 0:
            topology = FixedMixingMatrix(curr_topology, numberOfSampled)
            W_curr = topology(i)
            noise = np.random.normal(0, np.sqrt(sigma / num_dim), size=X_curr.shape)
            if shuffle:
              # shuffling 
              index, X_curr_sh, grad_curr_sh = shuffling(X_curr, grad_curr)
              # update
              X_temp_sh = X_curr_sh - gamma * (grad_curr_sh.T + noise)
              X_next_sh = X_temp_sh.dot(W_curr)
              # shuffling_back 
              X_next = shuffling_back(index, X_next_sh)
            else:
              X_temp = X_curr - gamma * (grad_curr.T + noise)
              X_next = X_temp.dot(W_curr)
            X_iter[:, sampledIndex] = X_next
          errors[curr_topology] += [consensus_distance(X_iter, A, B)]
            # print('X_next:', X_next)
    return errors, X_iter
