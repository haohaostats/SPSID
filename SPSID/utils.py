
# utils.py

import numpy as np
from config_simulation import EPS

# ==================== Simulation Utility Functions ====================

def simulate_direct_network(n, density):
    """Simulate a direct interaction network."""
    G = (np.random.rand(n, n) < density).astype(float)
    np.fill_diagonal(G, 0)
    return G

def simulate_observed_network(G_dir, beta=0.5, noise_level=0.1):
    """Simulate an observed network based on the direct network."""
    G_obs = G_dir + beta * (G_dir @ G_dir)
    G_obs = G_obs + noise_level * np.random.randn(*G_obs.shape)
    G_obs[G_obs < 0] = 0
    return np.nan_to_num(G_obs, nan=0.0)

def count_tp_edges(G_true, inferred_scores, k):
    """Count the number of true positives in the top k edges."""
    n_nodes = G_true.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)
    scores = inferred_scores[mask]
    true_edges = G_true[mask]
    top_k_indices = np.argsort(scores)[::-1][:k]
    return np.sum(true_edges[top_k_indices])