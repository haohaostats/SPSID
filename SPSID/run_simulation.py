
# run_simulation.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from config_simulation import *
from methods import spsid, rendor, nd_regulatory, network_enhancement, silencer, icm
from utils import simulate_direct_network, simulate_observed_network, count_tp_edges

def run_all_methods_on_simulation(density, beta, noise):
    """Runs all methods for a given set of parameters and returns performance scores."""
    G_dir = simulate_direct_network(N_NODES, density)
    G_obs = simulate_observed_network(G_dir, beta=beta, noise_level=noise)
    y_true = G_dir.flatten()

    scores = {
        'SPSID': spsid(G_obs, eps1=EPS1, eps2=EPS2, lambda_val=LAMBDA_VAL, return_tf_only=True).flatten(),
        'RENDOR': rendor(G_obs).flatten(),
        'ND': nd_regulatory(G_obs).flatten(),
        'NE': network_enhancement(G_obs).flatten(),
        'Silencer': silencer(G_obs).flatten(),
        'ICM': icm(G_obs).flatten()
    }
    
    results = {}
    for method, y_score in scores.items():
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        results[method] = {'AUROC': auroc, 'AUPR': aupr}
        
    return results

def run_basic_comparison():
    """Runs the basic performance comparison experiment."""
    print("--- Running Basic Performance Comparison ---")
    all_records = []
    for trial in range(NUM_TRIALS):
        trial_results = run_all_methods_on_simulation(
            density=BASIC_PARAMS['density'], 
            beta=BASIC_PARAMS['beta'], 
            noise=BASIC_PARAMS['noise']
        )
        for method, metrics in trial_results.items():
            record = {'Trial': trial + 1, 'Method': method, **metrics}
            all_records.append(record)
    
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(RESULTS_DIR, "performance_comparison_results.csv"), index=False)
    print("Basic performance comparison results saved.")

def run_sensitivity_analyses():
    """Runs sensitivity analyses for all parameters."""
    print("\n--- Running Sensitivity Analyses ---")
    param_configs = {
        'beta': (BETA_VALUES, {'density': SENSITIVITY_FIXED_PARAMS['density'], 'noise': SENSITIVITY_FIXED_PARAMS['noise']}),
        'noise': (SIGMA_VALUES, {'density': SENSITIVITY_FIXED_PARAMS['density'], 'beta': SENSITIVITY_FIXED_PARAMS['beta']}),
        'density': (DENSITY_VALUES, {'beta': SENSITIVITY_FIXED_PARAMS['beta'], 'noise': SENSITIVITY_FIXED_PARAMS['noise']})
    }

    for param_name, (param_values, fixed_params) in param_configs.items():
        all_records = []
        for val in param_values:
            for trial in range(NUM_TRIALS):
                current_params = fixed_params.copy()
                current_params[param_name] = val
                trial_results = run_all_methods_on_simulation(**current_params)
                for method, metrics in trial_results.items():
                    record = {param_name.capitalize(): val, 'Trial': trial + 1, 'Method': method, **metrics}
                    all_records.append(record)
        
        df = pd.DataFrame(all_records)
        df.to_csv(os.path.join(RESULTS_DIR, f"sensitivity_{param_name}_full_results.csv"), index=False)
        
        summary_df = df.groupby([param_name.capitalize(), 'Method'])[['AUROC', 'AUPR']].mean().reset_index()
        summary_df.to_csv(os.path.join(RESULTS_DIR, f"sensitivity_{param_name}_summary.csv"), index=False)
        print(f"Sensitivity analysis for '{param_name}' results saved.")

def run_tp_edge_comparison():
    """Runs the TP edge retention experiment."""
    print("\n--- Running TP Edge Comparison ---")
    tp_results = {m: np.zeros(len(EDGE_RANGE)) for m in METHODS}
    for trial in range(NUM_TRIALS):
        G_dir = simulate_direct_network(N_NODES, density=SENSITIVITY_FIXED_PARAMS['density'])
        G_obs = simulate_observed_network(G_dir, beta=SENSITIVITY_FIXED_PARAMS['beta'], noise_level=SENSITIVITY_FIXED_PARAMS['noise'])
        
        inferred = {
            'SPSID': spsid(G_obs, eps1=EPS1, eps2=EPS2, lambda_val=LAMBDA_VAL, return_tf_only=True),
            'RENDOR': rendor(G_obs),
            'ND': nd_regulatory(G_obs),
            'NE': network_enhancement(G_obs),
            'Silencer': silencer(G_obs),
            'ICM': icm(G_obs)
        }
        for i, k in enumerate(EDGE_RANGE):
            for m in METHODS:
                tp_results[m][i] += count_tp_edges(G_dir, inferred[m], k)
    
    for m in METHODS:
        tp_results[m] /= NUM_TRIALS

    records = [{'EdgesRetained': k, 'Method': m, 'AvgTP': tp_results[m][i]} 
               for m in METHODS for i, k in enumerate(EDGE_RANGE)]
    
    pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "tp_edges_comparison.csv"), index=False)
    print("TP edge comparison results saved.")

def run_lambda_sensitivity():
    """Runs the lambda parameter sensitivity analysis for SPSID."""
    print("\n--- Running Lambda Sensitivity Analysis for SPSID ---")
    lam_records = []
    for lam in LAMBDA_GRID:
        for trial in range(NUM_TRIALS):
            G_dir = simulate_direct_network(N_NODES, SENSITIVITY_FIXED_PARAMS['density'])
            G_obs = simulate_observed_network(G_dir, beta=SENSITIVITY_FIXED_PARAMS['beta'], noise_level=SENSITIVITY_FIXED_PARAMS['noise'])
            y_true = G_dir.ravel()
            y_score = spsid(G_obs, eps1=EPS1, eps2=EPS2, lambda_val=lam, return_tf_only=True).ravel()
            
            lam_records.append({
                'Lambda': lam, 'Trial': trial + 1,
                'AUROC': roc_auc_score(y_true, y_score),
                'AUPR': average_precision_score(y_true, y_score)
            })

    df = pd.DataFrame(lam_records)
    df.to_csv(os.path.join(RESULTS_DIR, "lambda_sensitivity_full_results.csv"), index=False)
    
    summary = df.groupby('Lambda').agg(
        Mean_AUROC=('AUROC', 'mean'), SD_AUROC=('AUROC', 'std'),
        Mean_AUPR=('AUPR', 'mean'), SD_AUPR=('AUPR', 'std'),
        N=('AUROC', 'size')
    ).reset_index()
    summary['SE_AUROC'] = summary['SD_AUROC'] / np.sqrt(summary['N'])
    summary['SE_AUPR'] = summary['SD_AUPR'] / np.sqrt(summary['N'])
    summary.to_csv(os.path.join(RESULTS_DIR, "lambda_sensitivity_summary.csv"), index=False)
    print("Lambda sensitivity analysis results saved.")


if __name__ == "__main__":
    np.random.seed(42)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    run_basic_comparison()
    run_sensitivity_analyses()
    run_tp_edge_comparison()
    run_lambda_sensitivity()
    print("\nAll experiments finished. Results saved to:", RESULTS_DIR)