
# utils_dream5.py

import os
import pandas as pd
import numpy as np

def load_gold_standard_edges(gs_file_path):
    """Loads a gold standard file and extracts true edges, TFs, and targets."""
    try:
        gs_df = pd.read_csv(gs_file_path, sep='\t', header=None, names=['TF', 'Target', 'Edge'])
        true_edges_df = gs_df[gs_df['Edge'] == 1].copy()
        gs_tfs = sorted(list(true_edges_df['TF'].unique()))
        gs_targets = sorted(list(true_edges_df['Target'].unique()))
        print(f"  Gold Standard: Loaded {len(gs_tfs)} TFs and {len(gs_targets)} Target Genes.")
        return true_edges_df, gs_tfs, gs_targets
    except Exception as e:
        print(f"  Error loading gold standard file {gs_file_path}: {e}")
        return None, None, None

def create_combined_matrix_from_df(all_tfs, all_targets, edges_df,
                                   tf_col='TF', target_col='Target', score_col=None,
                                   is_binary_truth=False, default_val=0.0):
    
    m_tfs = len(all_tfs)
    n_targets = len(all_targets)

    tf_to_idx = {tf: i for i, tf in enumerate(all_tfs)}
    target_to_idx = {target: i for i, target in enumerate(all_targets)}

    A = np.full((m_tfs, n_targets), default_val, dtype=np.float32)

    for _, row in edges_df.iterrows():
        tf = row[tf_col]
        target = row[target_col]
        if tf in tf_to_idx and target in target_to_idx:
            val = 1.0 if is_binary_truth else float(row[score_col])
            A[tf_to_idx[tf], target_to_idx[target]] = val

    N_combined = m_tfs + n_targets
    M = np.zeros((N_combined, N_combined), dtype=np.float32)
    M[:m_tfs, m_tfs:] = A
    M[m_tfs:, :m_tfs] = A.T
    return M