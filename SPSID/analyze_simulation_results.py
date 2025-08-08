

# analyze_simulation_results.py

import os
import pandas as pd
import numpy as np
from config_simulation import RESULTS_DIR, METHODS, NUM_TRIALS, LAMBDA_GRID

def analyze_basic_performance():
    """
    Reads raw performance results, calculates descriptive statistics for Table 1,
    and saves them to a CSV file.
    """
    print("--- Analyzing Basic Performance Comparison (for Table 1) ---")
    input_csv = os.path.join(RESULTS_DIR, "performance_comparison_results.csv")
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found. Run 'run_simulation.py' first.")
        return None

    df = pd.read_csv(input_csv)
    stats_data = []
    for method in METHODS:
        method_df = df[df['Method'] == method]
        if method_df.empty: continue
        
        stats = {"Method": method}
        for metric in ['AUROC', 'AUPR']:
            vals = method_df[metric]
            mean_val = np.mean(vals)
            se_val = np.std(vals, ddof=1) / np.sqrt(NUM_TRIALS)
            stats[f'Mean_{metric}'] = mean_val
            stats[f'Lower_CI_{metric}'] = mean_val - 1.96 * se_val
            stats[f'Upper_CI_{metric}'] = mean_val + 1.96 * se_val
        stats_data.append(stats)
        
    df_stats = pd.DataFrame(stats_data)
    output_csv = os.path.join(RESULTS_DIR, "descriptive_stats.csv")
    df_stats.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"Descriptive statistics saved to: {output_csv}\n")
    return df_stats

def process_sensitivity_results():
    """
    Processes all sensitivity analysis summaries into wide-format tables for Table 2.
    """
    print("--- Processing Sensitivity Analysis Results (for Table 2) ---")
    for param_name, part_letter in [('density', 'a'), ('noise', 'b'), ('beta', 'c')]:
        summary_csv = os.path.join(RESULTS_DIR, f"sensitivity_{param_name}_summary.csv")
        if not os.path.exists(summary_csv):
            print(f"WARNING: Summary file not found, skipping: {summary_csv}")
            continue
            
        df = pd.read_csv(summary_csv)
        pivot_df = df.pivot(index='Method', columns=param_name.capitalize(), values=['AUROC', 'AUPR']).reindex(METHODS)
        pivot_df.columns = [f'{metric}_{val}' for metric, val in pivot_df.columns]
        
        output_csv = os.path.join(RESULTS_DIR, f"table_part_{part_letter}_{param_name}.csv")
        pivot_df.round(3).to_csv(output_csv)
        print(f"Formatted table part ({part_letter}) for {param_name} saved to: {output_csv}")
    print("\n")

def analyze_lambda_sensitivity():
    """
    Calculates statistics for the lambda sensitivity analysis for Table 4.
    """
    print("--- Analyzing Lambda Sensitivity (for Table 4) ---")
    input_csv = os.path.join(RESULTS_DIR, "lambda_sensitivity_full_results.csv")
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found. Run 'run_simulation.py' first.")
        return None
        
    df = pd.read_csv(input_csv)
    summary_df = df.groupby('Lambda').agg(
        Mean_AUROC=('AUROC', 'mean'), SD_AUROC=('AUROC', 'std'),
        Mean_AUPR=('AUPR', 'mean'), SD_AUPR=('AUPR', 'std')
    ).reindex(LAMBDA_GRID)
    
    for metric in ['AUROC', 'AUPR']:
        summary_df[f'SE_{metric}'] = summary_df[f'SD_{metric}'] / np.sqrt(NUM_TRIALS)
        summary_df[f'Lower_CI_{metric}'] = summary_df[f'Mean_{metric}'] - 1.96 * summary_df[f'SE_{metric}']
        summary_df[f'Upper_CI_{metric}'] = summary_df[f'Mean_{metric}'] + 1.96 * summary_df[f'SE_{metric}']
        
    output_csv = os.path.join(RESULTS_DIR, "lambda_sensitivity_summary.csv")
    summary_df.to_csv(output_csv, float_format='%.5f')
    print(f"Lambda sensitivity statistics saved to: {output_csv}\n")
    return summary_df

if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        print("Please run 'run_simulation.py' first.")
    else:
        analyze_basic_performance()
        process_sensitivity_results()
        analyze_lambda_sensitivity()
        
        print("\nAll statistical analyses are complete. Summary CSV files have been updated/created.")
