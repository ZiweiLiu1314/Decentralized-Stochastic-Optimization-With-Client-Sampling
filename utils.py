import numpy as np


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

def return_step_size(parameters_tuple):
  """
  This function returns the step size suitable for each combination of parameters to produce the best results for each experiment
  These step sizes returned were tuned manually
  """

  # lists of value parameters we support
  list_of_topologies = ["ring", "centralized", "grid"]
  list_of_zetas  = [0, 1, 10]
  list_of_sigmas = [0, 1, 100]

  assert parameters_tuple[0] in list_of_topologies, f"toplogy should be inside {list_of_topologies}"
  assert parameters_tuple[1] in list_of_zetas, f"zeta should be inside {list_of_zetas}"
  assert parameters_tuple[2] in list_of_sigmas, f"sigma should be inside {list_of_sigmas}"

  step_size_mapper = {(list_of_topologies[0], list_of_zetas[0], list_of_sigmas[0]): 0.05,
                      (list_of_topologies[0], list_of_zetas[0], list_of_sigmas[1]): 0.003,
                      (list_of_topologies[0], list_of_zetas[0], list_of_sigmas[2]): 0.0001,
                      (list_of_topologies[0], list_of_zetas[1], list_of_sigmas[0]): 0.00012,
                      (list_of_topologies[0], list_of_zetas[1], list_of_sigmas[1]): 0.0001,
                      (list_of_topologies[0], list_of_zetas[1], list_of_sigmas[2]): 0.0001,
                      (list_of_topologies[0], list_of_zetas[2], list_of_sigmas[0]): 0.000065,
                      (list_of_topologies[0], list_of_zetas[2], list_of_sigmas[1]): 0.00005,
                      (list_of_topologies[0], list_of_zetas[2], list_of_sigmas[2]): 0.0001,
                      
                      (list_of_topologies[1], list_of_zetas[0], list_of_sigmas[0]): 0.05, 
                      (list_of_topologies[1], list_of_zetas[0], list_of_sigmas[1]): 0.007,
                      (list_of_topologies[1], list_of_zetas[0], list_of_sigmas[2]): 0.0001,
                      (list_of_topologies[1], list_of_zetas[1], list_of_sigmas[0]): 0.01,
                      (list_of_topologies[1], list_of_zetas[1], list_of_sigmas[1]): 0.005,
                      (list_of_topologies[1], list_of_zetas[1], list_of_sigmas[2]): 0.0001,
                      (list_of_topologies[1], list_of_zetas[2], list_of_sigmas[0]): 0.01,
                      (list_of_topologies[1], list_of_zetas[2], list_of_sigmas[1]): 0.003,
                      (list_of_topologies[1], list_of_zetas[2], list_of_sigmas[2]): 0.0001,

                      (list_of_topologies[2], list_of_zetas[0], list_of_sigmas[0]): 0.05, 
                      (list_of_topologies[2], list_of_zetas[0], list_of_sigmas[1]): 0.005,
                      (list_of_topologies[2], list_of_zetas[0], list_of_sigmas[2]): 0.0001,
                      (list_of_topologies[2], list_of_zetas[1], list_of_sigmas[0]): 0.001,
                      (list_of_topologies[2], list_of_zetas[1], list_of_sigmas[1]): 0.0015,
                      (list_of_topologies[2], list_of_zetas[1], list_of_sigmas[2]): 0.0001,
                      (list_of_topologies[2], list_of_zetas[2], list_of_sigmas[0]): 0.0005,
                      (list_of_topologies[2], list_of_zetas[2], list_of_sigmas[1]): 0.0005,
                      (list_of_topologies[2], list_of_zetas[2], list_of_sigmas[2]): 0.0001,
                      }

  return step_size_mapper[parameters_tuple]

  def sampling(thr, num_nodes, X, grad): #sampledIndex, X_curr
    # vector consisting of samples from uniform distribution
    sample = np.random.uniform(low = 0.0, high = 1.0, size = (num_nodes))
    sampledIndex = sample >= thr
    X_curr = X[:, sampledIndex]
    grad_curr = grad[sampledIndex, :]
    return sampledIndex, X_curr, grad_curr

def sampling_grid(thr, num_nodes, X, grad):
  # vector consisting of samples from uniform distribution
  sample = np.random.uniform(low = 0.0, high = 1.0, size = (num_nodes))
  sampledIndex = sample >= thr
  while (int(np.sqrt(sampledIndex.sum())) ** 2) != (sampledIndex.sum()):
    sample = np.random.uniform(low = 0.0, high = 1.0, size = (num_nodes))
    sampledIndex = sample >= thr
  X_curr = X[:, sampledIndex]
  grad_curr = grad[sampledIndex, :]
  return sampledIndex, X_curr, grad_curr

def sampling_s(n_sampled_nodes, n_total_nodes, X, grad):
  n_total_nodes_line = np.arange(n_total_nodes)
  choices = np.random.choice(n_total_nodes_line, n_sampled_nodes, replace=False)
  sampledIndex = np.array([False] * n_total_nodes)
  sampledIndex[choices] = True
  X_curr = X[:, sampledIndex]
  grad_curr = grad[sampledIndex, :]
  return sampledIndex, X_curr, grad_curr


def shuffling(X_curr, grad_curr):
  num_dim, num_sampled_nodes = X_curr.shape
  index = np.arange(num_sampled_nodes)
  np.random.shuffle(index)
  X_curr_sh = X_curr.T[index].T
  grad_curr_sh = grad_curr[index]
  return index, X_curr_sh, grad_curr_sh

def shuffling_back(index, X_curr_sh):
  X_curr_indexed = np.concatenate((index.reshape(1, len(index)), X_curr_sh), axis = 0)
  X_curr_indexed = X_curr_indexed[:, X_curr_indexed[0, :].argsort()]
  X_curr_indexed = X_curr_indexed[1:, :]
  return X_curr_indexed 