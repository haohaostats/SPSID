
# run_dream5_evaluation.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from config_dream5 import *
from methods import spsid, rendor, nd_regulatory, network_enhancement, silencer, icm
from utils_dream5 import load_gold_standard_edges, create_combined_matrix_from_df

def evaluate_one_observed_network(G_obs, y_true_flat):
    """
    Runs all denoising methods on a single observed network and returns performance.
    The logic and parameters here are identical to the original script.
    """
    results_list = []

    # --- Base (before denoising) ---
    y_score = G_obs.flatten()
    results_list.append({
        "Method": "Base",
        "AUROC": roc_auc_score(y_true_flat, y_score),
        "AUPR":  average_precision_score(y_true_flat, y_score)
    })

    # --- SPSID ---
    G_est = spsid(G_obs.copy(), eps1=1e-6, eps2=1e-6, lambda_val=1000, return_tf_only=True)
    y_score = G_est.flatten()
    results_list.append({
        "Method": "SPSID",
        "AUROC": roc_auc_score(y_true_flat, y_score),
        "AUPR":  average_precision_score(y_true_flat, y_score)
    })

    # --- Other competitor methods ---
    G_est_rendor = rendor(G_obs.copy(), m=2)
    results_list.append({
        "Method": "RENDOR",
        "AUROC": roc_auc_score(y_true_flat, G_est_rendor.flatten()),
        "AUPR":  average_precision_score(y_true_flat, G_est_rendor.flatten())
    })
    
    G_est_nd = nd_regulatory(G_obs.copy(), beta=0.5, alpha=1.0)
    results_list.append({
        "Method": "ND",
        "AUROC": roc_auc_score(y_true_flat, G_est_nd.real.flatten()),
        "AUPR":  average_precision_score(y_true_flat, G_est_nd.real.flatten())
    })

    G_est_ne = network_enhancement(G_obs.copy(), order=2, K=20, alpha=0.9)
    results_list.append({
        "Method": "NE",
        "AUROC": roc_auc_score(y_true_flat, G_est_ne.flatten()),
        "AUPR":  average_precision_score(y_true_flat, G_est_ne.flatten())
    })

    G_est_sil = silencer(G_obs.copy())
    results_list.append({
        "Method": "Silencer",
        "AUROC": roc_auc_score(y_true_flat, G_est_sil.flatten()),
        "AUPR":  average_precision_score(y_true_flat, G_est_sil.flatten())
    })

    G_est_icm = icm(G_obs.copy(), lambda_icm=1e-2)
    results_list.append({
        "Method": "ICM",
        "AUROC": roc_auc_score(y_true_flat, G_est_icm.flatten()),
        "AUPR":  average_precision_score(y_true_flat, G_est_icm.flatten())
    })

    return results_list

def main():
    """
    Main function to orchestrate the DREAM5 evaluation.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    all_results = []
    
    for net_id, gs_filename in NETWORK_INFO.items():
        print(f"\n================= PROCESSING NETWORK {net_id} =================\n")
        
        # 1. Load Gold Standard for the current network
        gs_path = os.path.join(GOLD_STANDARD_DIR, gs_filename)
        gs_edges, gs_tfs, gs_targets = load_gold_standard_edges(gs_path)
        if gs_edges is None:
            continue
        
        M_true = create_combined_matrix_from_df(gs_tfs, gs_targets, gs_edges, is_binary_truth=True)
        y_true_flat = M_true.flatten()

        # 2. Iterate through the 4 inference methods for this network
        for inf_method in INFERENCE_METHODS:
            obs_filename = f"net{net_id}_inferred_{inf_method}.tsv"
            obs_path = os.path.join(GRN_NETWORK_DIR, obs_filename)
            
            if not os.path.exists(obs_path):
                print(f"[WARNING] Input network file not found, skipping: {obs_path}")
                continue
            
            print(f"--- Evaluating Denoising on: {obs_filename} ---")
            obs_df = pd.read_csv(obs_path, sep='\t')
            M_obs = create_combined_matrix_from_df(gs_tfs, gs_targets, obs_df, score_col='Score')

            # 3. Evaluate all 7 denoising methods on this observed network

            results_for_one_run = evaluate_one_observed_network(M_obs.copy(), y_true_flat)
            
            for r in results_for_one_run:
                r["NetworkID"] = net_id
                r["InputInference"] = inf_method
                all_results.append(r)

    # 4. Compile and save the final results dataframe
    df_all = pd.DataFrame(all_results)
    df_all = df_all[["NetworkID", "InputInference", "Method", "AUROC", "AUPR"]]
    
    output_csv = os.path.join(RESULTS_DIR, "dream5_all_performance_results.csv")
    df_all.to_csv(output_csv, index=False, float_format='%.5f')

    print("\n" + "="*50)
    print("=== All evaluations finished successfully! ===")
    print("Full performance results saved to:", output_csv)
    print("="*50)

if __name__ == '__main__':
    main()