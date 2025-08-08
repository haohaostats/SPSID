
# plot_simulation_results.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config_simulation import RESULTS_DIR, METHODS, COLOR_MAP

def plot_basic_comparison_charts():
    """
    Plots a combined 2x2 figure with boxplots and bar charts 
    for the basic performance comparison, with labels (A), (B), (C), (D)
    in the top-left corner of each subplot.
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "performance_comparison_results.csv"))
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    auroc_data = [df[df['Method'] == m]['AUROC'] for m in METHODS]
    aupr_data = [df[df['Method'] == m]['AUPR'] for m in METHODS]
    summary = df.groupby('Method')[['AUROC', 'AUPR']].mean().reindex(METHODS)
    x = np.arange(len(METHODS))

    # --- Subplot (A): AUROC Distribution (Box Plot) ---
    ax_A = axs[0, 0]
    bplot1 = ax_A.boxplot(auroc_data, labels=METHODS, patch_artist=True, widths=0.6)
    for patch in bplot1['boxes']:
        patch.set_facecolor("skyblue")
    ax_A.set_title("AUROC Distribution Across Methods", fontweight='bold', fontsize=14)
    ax_A.set_ylabel("AUROC")
    ax_A.tick_params(axis='x', rotation=45)
    ax_A.grid(True, linestyle='--', alpha=0.6)
    ax_A.text(0, 1.1, '(A)', transform=ax_A.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # --- Subplot (B): AUPR Distribution (Box Plot) ---
    ax_B = axs[0, 1]
    bplot2 = ax_B.boxplot(aupr_data, labels=METHODS, patch_artist=True, widths=0.6)
    for patch in bplot2['boxes']:
        patch.set_facecolor("lightgreen")
    ax_B.set_title("AUPR Distribution Across Methods", fontweight='bold', fontsize=14)
    ax_B.set_ylabel("AUPR")
    ax_B.tick_params(axis='x', rotation=45)
    ax_B.grid(True, linestyle='--', alpha=0.6)
    ax_B.text(0, 1.1, '(B)', transform=ax_B.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # --- Subplot (C): Mean AUROC (Bar Chart) ---
    ax_C = axs[1, 0]
    ax_C.bar(x, summary['AUROC'], color=[COLOR_MAP.get(m, 'gray') for m in METHODS])
    ax_C.set_xticks(x)
    ax_C.set_xticklabels(METHODS, rotation=45, ha='right')
    ax_C.set_title("Mean AUROC", fontweight='bold', fontsize=14)
    ax_C.set_ylabel("AUROC")
    ax_C.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax_C.text(0, 1.1, '(C)', transform=ax_C.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # --- Subplot (D): Mean AUPR (Bar Chart) ---
    ax_D = axs[1, 1]
    ax_D.bar(x, summary['AUPR'], color=[COLOR_MAP.get(m, 'gray') for m in METHODS])
    ax_D.set_xticks(x)
    ax_D.set_xticklabels(METHODS, rotation=45, ha='right')
    ax_D.set_title("Mean AUPR", fontweight='bold', fontsize=14)
    ax_D.set_ylabel("AUPR")
    ax_D.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax_D.text(0, 1.1, '(D)', transform=ax_D.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # --- Final Touches and Saving ---
    # fig.suptitle("Performance Comparison of Denoising Methods", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    plt.savefig(os.path.join(RESULTS_DIR, "Figure_2.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "Figure_2.eps"), format='eps')
    plt.close(fig)

    print("Combined 2x2 comparison plot saved as 'performance_summary_plot.[png/eps]'.")

def plot_sensitivity_curves():
    """Plots sensitivity analysis curves for all parameters."""
    params = {
        'beta': "Transitive Effect Strength (beta)",
        'noise': "Noise Level (sigma)",
        'density': "Network Density (rho)"
    }
    for param_name, xlabel in params.items():
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"sensitivity_{param_name}_summary.csv"))
        param_cap = param_name.capitalize()
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        for m in METHODS:
            sub_df = df[df["Method"] == m]
            axs[0].plot(sub_df[param_cap], sub_df["AUROC"], marker='o', label=m, color=COLOR_MAP[m])
            axs[1].plot(sub_df[param_cap], sub_df["AUPR"], marker='o', label=m, color=COLOR_MAP[m])

        axs[0].set_title(f"AUROC vs. {param_cap}")
        axs[1].set_title(f"AUPR vs. {param_cap}")
        for ax in axs:
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean Score")
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(f"Performance vs. {param_cap}", fontsize=16)
        fig.legend(METHODS, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(METHODS))
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(RESULTS_DIR, f"sensitivity_{param_name}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(RESULTS_DIR, f"sensitivity_{param_name}.eps"), format="eps", bbox_inches="tight")
        plt.close(fig)

    print(f"Sensitivity analysis plots for {list(params.keys())} saved.")

def plot_tp_edge_curve():
    """Plots the TP edge retention curve."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "tp_edges_comparison.csv"))
    
    plt.figure(figsize=(10, 6))
    for m in METHODS:
        sub_df = df[df['Method'] == m]
        plt.plot(sub_df['EdgesRetained'], sub_df['AvgTP'], marker='o', label=m, color=COLOR_MAP[m])
    
    plt.xlabel("Number of Edges Retained (k)")
    plt.ylabel("Average Number of True Positive Edges")
    plt.title("TP Edge Retention", fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Figure_3.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "Figure_3.eps"), format='eps')
    plt.close()

    print("TP edge comparison plot saved.")

def plot_lambda_sensitivity_curve():
    """Plots the lambda parameter sensitivity curve for SPSID."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "lambda_sensitivity_summary.csv"))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(df['Lambda'], df['Mean_AUROC'], yerr=1.96 * df['SE_AUROC'], fmt="-o", capsize=3, color='blue')
    axes[0].set_xscale('log')
    axes[0].set_title('AUROC vs. Lambda')
    
    axes[1].errorbar(df['Lambda'], df['Mean_AUPR'], yerr=1.96 * df['SE_AUPR'], fmt="-o", capsize=3, color='green')
    axes[1].set_xscale('log')
    axes[1].set_title('AUPR vs. Lambda')
    
    for ax in axes:
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Score')
        ax.grid(True, which="both", ls="--")

    plt.suptitle("SPSID Sensitivity to Lambda Parameter", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(os.path.join(RESULTS_DIR, "lambda_sensitivity.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "lambda_sensitivity.eps"), format='eps')
    plt.close(fig)

    print("Lambda sensitivity plot saved.")

if __name__ == "__main__":
    # Ensure the results directory exists, as this script might be run independently
    if not os.path.exists(RESULTS_DIR):
        print(f"Warning: Results directory not found at {RESULTS_DIR}. Creating it.")
        os.makedirs(RESULTS_DIR)

    plot_basic_comparison_charts()
    # plot_sensitivity_curves()
    plot_tp_edge_curve()
    # plot_lambda_sensitivity_curve()
    print("\nAll simulation plots generated. Check the results directory:", RESULTS_DIR)