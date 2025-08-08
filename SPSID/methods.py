
# methods.py

import numpy as np
from scipy.linalg import null_space

# ==================== Global Settings ====================
EPS = 1e-8

# ==================== Core Algorithm Implementations ====================

def spsid(W_obs, eps1=1e-6, eps2=1e-6, lambda_val=1000, return_tf_only=True):
    """SPSID algorithm implementation."""
    EPS_internal = 1e-12

    W_obs = np.asarray(W_obs)
    n_tf, n_total = W_obs.shape
    
    if n_tf != n_total:
        pad = np.hstack((np.zeros((n_total - n_tf, n_tf)), np.eye(n_total - n_tf)))
        W_sq = np.vstack((W_obs, pad))
    else:
        W_sq = W_obs.copy()
    
    n_nodes = W_sq.shape[0]
    J, I = np.ones((n_nodes, n_nodes)), np.eye(n_nodes)

    W_tilde = W_sq + eps1 * J + eps2 * I
    row_sum = W_tilde.sum(axis=1, keepdims=True) + EPS_internal
    P_obs = W_tilde / row_sum

    A = I + P_obs + lambda_val * I
    P_dir = P_obs @ np.linalg.solve(A, I)

    eigvals, eigvecs = np.linalg.eig(P_obs.T)
    idx1 = np.argmin(np.abs(eigvals - 1))
    pi = np.abs(eigvecs[:, idx1].real)
    pi /= pi.sum()

    W_dir = np.diag(pi) @ P_dir
    W_dir = np.nan_to_num(W_dir, nan=0.0)
    
    if return_tf_only and n_tf != n_total:
        return W_dir[:n_tf, :]
    else:
        return W_dir

def _safe_inv(mat, *, rcond=1e-12):
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat, rcond=rcond)

def _safe_null_space(A, *, rcond=1e-12):
    try:
        return null_space(A, rcond=rcond)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eig(A)
        idx = np.argmin(np.abs(w))
        return v[:, idx:idx + 1]

def rendor(mat: np.ndarray, m: int = 2) -> np.ndarray:
    n_tf, n_nodes = mat.shape
    mat = mat.astype(float, copy=True)
    np.fill_diagonal(mat[:n_tf, :n_tf], 0.0)
    tf_block = mat[:n_tf, :n_tf]
    mat[:n_tf, :n_tf] = 0.5 * (tf_block + tf_block.T)

    if n_nodes > n_tf:
        lower_block = np.zeros((n_nodes - n_tf, n_nodes))
        np.fill_diagonal(lower_block[:, n_tf:], 1)
        mat1 = np.vstack([mat, lower_block])
    else:
        mat1 = mat.copy()
    
    mat1 = 0.5 * (mat1 + mat1.T)
    mn, mx = mat1.min(), mat1.max()
    mat1 = (mat1 - mn) / (mx - mn + EPS)
    positive_vals = mat1[mat1 > 0]
    eps_val = positive_vals.min() if positive_vals.size else EPS
    mat1 += eps_val
    mat1 += eps_val * np.eye(mat1.shape[0])
    row_sums = mat1.sum(axis=1, keepdims=True) + EPS
    P1 = mat1 / row_sums
    I = np.eye(mat1.shape[0])
    inv_term = _safe_inv((m - 1) * I + P1)
    P2 = m * P1 @ inv_term

    col_mins = P2.min(axis=0)
    P2 -= np.minimum(col_mins, 0.0)
    P2 /= (P2.sum(axis=1, keepdims=True) + EPS)
    ns = _safe_null_space((P2 - I).T)
    stat_d = np.abs(ns[:, 0]) if ns.size > 0 else np.ones(P2.shape[0])
    stat_d /= (stat_d.sum() + EPS)

    net_new = np.diag(stat_d) @ P2
    net_new = net_new + net_new.T
    return np.nan_to_num(net_new[:n_tf], nan=0.0)

def nd_regulatory(mat, *, beta=0.5, alpha=1.0, eps=1e-12, random_state=1):
    mat = mat.astype(float).copy()
    n_tf, n_nodes = mat.shape

    diag_idx = np.arange(min(n_tf, n_nodes))
    mat[diag_idx, diag_idx] = 0.0

    tf_block = mat[:n_tf, :n_tf]
    tf_sym = tf_block.copy()
    for i in range(n_tf):
        for j in range(i + 1, n_tf):
            v1, v2 = tf_block[i, j], tf_block[j, i]
            val = 0.5 * (v1 + v2) if v1 and v2 else v2 if v1 == 0 else v1
            tf_sym[i, j] = tf_sym[j, i] = val
    mat[:n_tf, :n_tf] = tf_sym

    mat_th = mat * (mat >= np.quantile(mat, 1.0 - alpha)) if alpha < 1.0 else mat.copy()
    
    mat_th[:n_tf, :n_tf] = 0.5 * (mat_th[:n_tf, :n_tf] + mat_th[:n_tf, :n_tf].T)

    temp_net = (mat_th > 0.0).astype(float)
    mat_th_remain = mat * (1.0 - temp_net)
    m11 = mat_th_remain.max(initial=0.0)

    mat1 = np.vstack((mat_th, np.zeros((n_nodes - n_tf, n_nodes)))) if n_nodes > n_tf else mat_th.copy()

    def _eig_with_check(A):
        eigvals, U = np.linalg.eig(A)
        try:
            condU = np.linalg.cond(U)
        except np.linalg.LinAlgError:
            condU = np.inf
        return eigvals, U, condU

    eigvals, U, condU = _eig_with_check(mat1)
    if condU > 1.0e10:
        rng = np.random.default_rng(seed=random_state)
        r_p = 1e-3
        rand_tf = rng.random((n_tf, n_tf)) * r_p
        rand_tf = 0.5 * (rand_tf + rand_tf.T)
        np.fill_diagonal(rand_tf, 0.0)
        rand_target = rng.random((n_tf, n_nodes - n_tf)) * r_p if n_nodes > n_tf else np.empty((n_tf, 0))
        mat_th += np.hstack((rand_tf, rand_target))
        mat1 = np.vstack((mat_th, np.zeros((n_nodes - n_tf, n_nodes)))) if n_nodes > n_tf else mat_th.copy()
        eigvals, U, _ = _eig_with_check(mat1)

    lam_n = max(0.0, -eigvals.real.min())
    lam_p = max(0.0, eigvals.real.max())
    m1 = lam_p * (1.0 - beta) / beta
    m2 = lam_n * (1.0 + beta) / beta
    scale = max(m1, m2, eps)
    eig_scaled = eigvals / (scale + eigvals)
    D_scaled = np.diag(eig_scaled)

    try:
        U_inv = np.linalg.inv(U)
    except np.linalg.LinAlgError:
        U_inv = np.linalg.pinv(U)
    net_new = U @ D_scaled @ U_inv
    net_new2 = net_new[:n_tf, :]
    m2_val = net_new2.min(initial=0.0)
    net_new3 = (net_new2 + max(m11 - m2_val, 0.0)) * temp_net
    mat_nd = net_new3 + mat_th_remain
    return np.nan_to_num(mat_nd, nan=0.0)

def network_enhancement(W_in, order=2, K=20, alpha=0.9):
    n_nodes = W_in.shape[0]
    W_in1 = W_in * (1 - np.eye(n_nodes))
    zeroindex = np.where(np.sum(np.abs(W_in1), axis=0) > 0)[0]
    
    if zeroindex.size == 0:
        return np.zeros_like(W_in)
        
    W0 = W_in[np.ix_(zeroindex, zeroindex)]
    W = _ne_dn(W0, mode='ave')
    W = 0.5 * (W + W.T)
    DD = np.sum(np.abs(W0), axis=0)
    P = _dominateset(np.abs(W), min(K, len(W) - 1)) * np.sign(W) if np.unique(W).size > 2 else W.copy()
    
    P += np.eye(len(P)) + np.diag(np.sum(np.abs(P), axis=1))
    P = _transition_fields(P)
    d, U = np.linalg.eigh(P)
    d = (1 - alpha) * d / (1 - alpha * (d ** order) + EPS)
    W_enh = (U * d) @ U.T
    
    W_enh = (W_enh * (1 - np.eye(len(W_enh)))) / (1 - np.diag(W_enh) + EPS)[:, None]
    W_enh = DD[:, None] * W_enh
    W_enh[W_enh < 0] = 0
    W_enh = 0.5 * (W_enh + W_enh.T)
    W_out = np.zeros_like(W_in)
    W_out[np.ix_(zeroindex, zeroindex)] = W_enh
    return W_out

def _ne_dn(w, mode='ave'):
    n_nodes = len(w)
    D = np.sum(np.abs(w * n_nodes), axis=1) + EPS
    Dinv = np.diag(1.0 / D) if mode == 'ave' else np.diag(1.0 / np.sqrt(D))
    return Dinv @ (w * n_nodes) if mode == 'ave' else Dinv @ (w * n_nodes @ Dinv)

def _transition_fields(W):
    zeroindex = np.where(np.sum(W, axis=1) == 0)[0]
    W = _ne_dn(W * len(W), mode='ave')
    w = np.sqrt(np.sum(np.abs(W), axis=0) + EPS)
    W = W / w[None, :]
    W = W @ W.T
    W[zeroindex, :] = 0
    W[:, zeroindex] = 0
    return W

def _dominateset(aff, k):
    n_nodes = aff.shape[0]
    idx = np.argsort(aff, axis=1)[:, ::-1]
    mask = np.zeros_like(aff, dtype=bool)
    mask[np.arange(n_nodes)[:, None], idx[:, :k]] = True
    P = np.zeros_like(aff)
    P[mask] = aff[mask]
    return 0.5 * (P + P.T)

def silencer(C, *, ridge=1e-6):
    C = C.astype(float)
    n_nodes = C.shape[0]
    I = np.eye(n_nodes)
    numer = (C - I) + np.diag(np.diag((C - I) @ C))
    S = numer @ np.linalg.inv(C + ridge * I)
    return np.nan_to_num(0.5 * (S + S.T), nan=0.0)

def icm(mat, lambda_icm=1e-2):
    C = 0.5 * (mat + mat.T)
    precision = np.linalg.inv(C + lambda_icm * np.eye(C.shape[0]))
    d = np.sqrt(np.clip(np.diag(precision), a_min=EPS, a_max=None))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1)
    return np.nan_to_num(0.5 * (partial_corr + partial_corr.T), nan=0.0)