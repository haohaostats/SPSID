
# plot_dream5_results.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from config_dream5 import RESULTS_DIR, ALL_METHODS, INFERENCE_METHODS

# ==================== Main Plotting Functions ====================

def plot_facet_heatmap():

    df = pd.read_csv(os.path.join(RESULTS_DIR, "dream5_all_performance_results.csv"))

    methods_for_plot = [m for m in ALL_METHODS if m != 'Base']
    df = df[df['Method'].isin(methods_for_plot)]
    
    df['Method'] = pd.Categorical(df['Method'], categories=methods_for_plot, ordered=True)
    
    df_pivot = df.pivot_table(
        index=['InputInference', 'NetworkID'], 
        columns='Method', 
        values=['AUROC', 'AUPR']
    )

    network_labels = {1: 'Network 1', 2: 'Network 2', 3: 'Network 3', 4: 'Network 4'}
    inference_labels = {
        'pearson': 'Pearson', 'spearman': 'Spearman', 
        'grnboost': 'GRNBoost2', 'genie3': 'GENIE3'
    }

    fig, axes = plt.subplots(len(INFERENCE_METHODS), 2, figsize=(16, 11), sharex='col')
    
    mappable_auroc = None
    mappable_aupr = None

    metrics_config = {
        'AUROC': {'cmap': 'viridis', 'vmin': 0.7, 'vmax': 1.0},
        'AUPR': {'cmap': 'plasma', 'vmin': 0.0, 'vmax': 0.4}
    }

    for i, inference_method in enumerate(INFERENCE_METHODS):
        for j, metric in enumerate(['AUROC', 'AUPR']):
            ax = axes[i, j]
            
            subset = df_pivot.loc[inference_method, metric]
            subset = subset.rename(index=network_labels)

            heatmap = sns.heatmap(
                subset, ax=ax, annot=True, fmt=".3f", 
                cmap=metrics_config[metric]['cmap'],
                vmin=metrics_config[metric]['vmin'],
                vmax=metrics_config[metric]['vmax'],
                cbar=False, 
                linewidths=.5, annot_kws={"size": 10}
            )
            if i == 0 and j == 0: mappable_auroc = heatmap.collections[0]
            if i == 0 and j == 1: mappable_aupr = heatmap.collections[0]
            
            if i == len(INFERENCE_METHODS) - 1:
                ax.set_xlabel('Denoising Method', fontsize=14)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
            else:
                ax.set_xlabel('')
            
            if j == 0:
                ax.set_ylabel(inference_labels[inference_method], fontsize=14, fontweight='bold')
            
            if i == 0:
                panel_label = f"({chr(65+j)})"

                ax.text(-0.15, 1.18, panel_label, transform=ax.transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='left')
                
                ax.set_title(metric, fontsize=16, fontweight='bold', pad=30)
            
            if j > 0:
                ax.set_ylabel('')

            ax.tick_params(axis='y', rotation=0, labelsize=12)

    fig.subplots_adjust(right=0.85)
    cbar_ax_auroc = fig.add_axes([0.88, 0.53, 0.02, 0.35])
    cbar_auroc = fig.colorbar(mappable_auroc, cax=cbar_ax_auroc)
    cbar_auroc.set_label('AUROC Score', fontsize=12)
    
    cbar_ax_aupr = fig.add_axes([0.88, 0.12, 0.02, 0.35])
    cbar_aupr = fig.colorbar(mappable_aupr, cax=cbar_ax_aupr)
    cbar_aupr.set_label('AUPR Score', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.87, 0.96])
    
    filepath = os.path.join(RESULTS_DIR, 'Figure_4')
    plt.savefig(f"{filepath}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filepath}.eps", format='eps', bbox_inches='tight')
    plt.close(fig)
    
def plot_performance_distributions():

    df = pd.read_csv(os.path.join(RESULTS_DIR, "dream5_all_performance_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, (ax, metric) in enumerate(zip(axes, ['AUROC', 'AUPR'])):
        
        sns.boxplot(x='Method', y=metric, data=df, ax=ax, order=ALL_METHODS,
                    showfliers=False,
                    width=0.5,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))

        sns.stripplot(x='Method', y=metric, data=df, ax=ax, order=ALL_METHODS, 
                      hue='Method', palette='Set2',
                      size=5, jitter=True, legend=False)

        means = df.groupby('Method')[metric].mean().reindex(ALL_METHODS)
        
        for method_idx, method in enumerate(ALL_METHODS):
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(method_idx, means[method] + y_offset, f'{means[method]:.3f}', 
                    ha='center', 
                    color='black', 
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        panel_label = f"({chr(65+i)})"
        ax.text(-0.09, 1.09, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
        
        ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout(pad=3.0, w_pad=4.0) 
    
    filepath = os.path.join(RESULTS_DIR, 'Figure_5') 
    plt.savefig(f"{filepath}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filepath}.eps", format='eps', bbox_inches='tight')
    plt.close(fig)


def plot_combined_improvement_analysis():

    FONT_SIZES = {
        'panel_label': 20, 'main_title': 16, 'axis_title': 14,
        'tick_label': 14, 'legend_title': 14, 'legend_item': 14,
        'heatmap_group': 14, 'heatmap_annot': 12
    }
    
    inference_labels = {
        'pearson': 'Pearson', 'spearman': 'Spearman', 
        'grnboost': 'GRNBoost2', 'genie3': 'GENIE3'
    }

    try:
        df_perf = pd.read_csv(os.path.join(RESULTS_DIR, "dream5_all_performance_results.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure CSV file is in the '{RESULTS_DIR}' directory.")
        return

    network_map = {1: 'Network 1', 2: 'Network 2', 3: 'Network 3', 4: 'Network 4'}
    df_perf['NetworkName'] = df_perf['NetworkID'].map(network_map)
        
    base_scores = df_perf[df_perf['Method'] == 'Base'].set_index(['NetworkID', 'NetworkName', 'InputInference'])
    denoised_scores = df_perf[df_perf['Method'] != 'Base']
    merged = denoised_scores.merge(base_scores, on=['NetworkID', 'NetworkName', 'InputInference'], suffixes=('', '_base'))
    
    merged['AUROC_Improvement_Percent'] = (merged['AUROC'] - merged['AUROC_base']) / merged['AUROC_base'] * 100
    merged['AUPR_Improvement_Percent'] = (merged['AUPR'] - merged['AUPR_base']) / merged['AUPR_base'] * 100
    merged['Delta_AUROC'] = merged['AUROC'] - merged['AUROC_base']
    merged['Delta_AUPR'] = merged['AUPR'] - merged['AUPR_base']

    fig = plt.figure(figsize=(18, 16)) 
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], wspace=0.3, hspace=0.25) 
    ax_heat_auroc = fig.add_subplot(gs[0, 0])
    ax_heat_aupr = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, :])

    # --- (A) & (B) Heatmaps ---
    methods_no_base = [m for m in ALL_METHODS if m != 'Base']
    merged['InputInference'] = pd.Categorical(merged['InputInference'], categories=INFERENCE_METHODS, ordered=True)
    merged['Method'] = pd.Categorical(merged['Method'], categories=methods_no_base, ordered=True)

    for ax, metric in [(ax_heat_auroc, 'AUROC_Improvement_Percent'), (ax_heat_aupr, 'AUPR_Improvement_Percent')]:
        pivot_data = merged.pivot_table(index=['InputInference', 'Method'], columns='NetworkName', values=metric)
        sns.heatmap(pivot_data, ax=ax, cmap='RdBu_r', center=0, annot=True, fmt='.1f', linewidths=.5, annot_kws={"size": FONT_SIZES['heatmap_annot']})
        
        ax.set_yticks([(i + 0.5) for i in range(len(pivot_data))])
        ax.set_yticklabels(pivot_data.index.get_level_values('Method'), rotation=0, fontsize=FONT_SIZES['tick_label'])
        ax.tick_params(axis='y', length=0)
        ax.set_ylabel('')

        y_pos = 0
        for i, (inf_method, grp) in enumerate(pivot_data.groupby(level='InputInference')):
            display_name = inference_labels.get(inf_method, inf_method.capitalize())
            ax.text(-0.9, y_pos + grp.shape[0] / 2, display_name, 
                    ha='center', va='center', rotation=90, 
                    fontsize=FONT_SIZES['heatmap_group'], fontweight='bold')
            y_pos += grp.shape[0]
            if i < len(pivot_data.index.levels[0]) - 1:
                ax.axhline(y_pos, color='black', linewidth=1.5)
        
        ax.set_xlabel('Network', fontsize=FONT_SIZES['axis_title'])
        ax.tick_params(axis='x', rotation=0, labelsize=FONT_SIZES['tick_label'])
    
    ax_heat_auroc.set_title('AUROC % Improvement over Base', fontsize=FONT_SIZES['main_title'], fontweight='bold')
    ax_heat_auroc.text(-0.35, 1.05, '(A)', transform=ax_heat_auroc.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')
    
    ax_heat_aupr.set_title('AUPR % Improvement over Base', fontsize=FONT_SIZES['main_title'], fontweight='bold')
    ax_heat_aupr.text(-0.35, 1.05, '(B)', transform=ax_heat_aupr.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')

    # --- (C) Scatter Plot ---
    palette = "Set1"
    input_markers = {'genie3': 'o', 'grnboost': 's', 'pearson': '^', 'spearman': 'D'}
    
    sns.scatterplot(data=merged, x='Delta_AUROC', y='Delta_AUPR', hue='Method', style='InputInference',
                    hue_order=methods_no_base, style_order=INFERENCE_METHODS, palette=palette, markers=input_markers,
                    s=120, edgecolor='black', linewidth=0.8, ax=ax_scatter)

    ax_scatter.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_scatter.set_axisbelow(True)
    ax_scatter.set_xscale('symlog', linthresh=1e-4)
    ax_scatter.set_yscale('symlog', linthresh=1e-4)
    ax_scatter.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_scatter.axvline(0, color='gray', linestyle='--', linewidth=1)
    
    ax_scatter.set_title('ΔAUROC vs ΔAUPR Across All Methods and Inferences (Symlog)', fontsize=FONT_SIZES['main_title'], fontweight='bold')
    ax_scatter.set_xlabel('ΔAUROC (symlog scale)', fontsize=FONT_SIZES['axis_title'])
    ax_scatter.set_ylabel('ΔAUPR (symlog scale)', fontsize=FONT_SIZES['axis_title'])
    ax_scatter.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick_label'])
    ax_scatter.text(-0.05, 1.05, '(C)', transform=ax_scatter.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')
    
    legend = ax_scatter.get_legend()
    legend.set_bbox_to_anchor((1.02, 1))
    legend.set_loc('upper left')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    legend.set_title(None)

    all_texts = legend.texts
    split_point = 1 + len(methods_no_base) 

    all_texts[0].set_text('Method')
    all_texts[0].set_fontsize(FONT_SIZES['legend_title'])
    all_texts[0].set_weight('bold')
    for txt in all_texts[1:split_point]:
        txt.set_fontsize(FONT_SIZES['legend_item'])

    all_texts[split_point].set_text('Inference')
    all_texts[split_point].set_fontsize(FONT_SIZES['legend_title'])
    all_texts[split_point].set_weight('bold')

    for i, txt in enumerate(all_texts[split_point + 1:]):
        original_label = INFERENCE_METHODS[i]
        txt.set_text(inference_labels.get(original_label, original_label))
        txt.set_fontsize(FONT_SIZES['legend_item'])

    fig.tight_layout(pad=3.0)
    
    filepath = os.path.join(RESULTS_DIR, 'Figure_6')
    plt.savefig(f"{filepath}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filepath}.eps", format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"Combined improvement analysis plot saved to {filepath}.[png/eps]")


def plot_combined_rank_analysis():

    FONT_SIZES = {
        'panel_label': 20,
        'main_title': 18,
        'axis_title': 16,
        'tick_label': 14,
        'legend_title': 16,
        'legend_item': 14,
        'lollipop_label': 12
    }
    
    inference_labels = {
        'pearson': 'Pearson', 'spearman': 'Spearman', 
        'grnboost': 'GRNBoost2', 'genie3': 'GENIE3'
    }

    try:
        df_rs = pd.read_csv(os.path.join(RESULTS_DIR, "dream5_rankscore_details.csv"))
        df_friedman = pd.read_csv(os.path.join(RESULTS_DIR, "friedman_nemenyi_summary.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure CSV files are in the '{RESULTS_DIR}' directory.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(18, 14)) 
    panel_labels = ['(A)', '(B)', '(C)', '(D)']
    metrics = ['AUROC', 'AUPR']

    for col_idx, metric in enumerate(metrics):
        
        # --- Top Row: Bar Charts ---
        ax_bar = axes[0, col_idx]
        sub_rs = df_rs[df_rs['Metric'] == metric]
        mean_tbl = sub_rs.pivot_table(index="Method", columns="InputInference", values="RankScore").reindex(index=ALL_METHODS, columns=INFERENCE_METHODS)
        err_tbl = sub_rs.pivot_table(index="Method", columns="InputInference", values="RankScore", aggfunc=lambda x: (x.quantile(0.75) - x.quantile(0.25)) / 2).reindex_like(mean_tbl)

        x_base = np.arange(len(ALL_METHODS))
        bar_w = 0.18
        colors = ["#4c72b0", "#dd8452", "#c44e52", "#55a868"]

        for j, eng in enumerate(INFERENCE_METHODS):
            xs = x_base + (j - (len(INFERENCE_METHODS) - 1) / 2) * bar_w
            label = inference_labels.get(eng, eng.capitalize())
            ax_bar.bar(xs, mean_tbl[eng], width=bar_w, color=colors[j], edgecolor='k', linewidth=0.4, alpha=0.85, label=label)
            ax_bar.errorbar(xs, mean_tbl[eng], yerr=err_tbl[eng], fmt='none', ecolor='k', capsize=3, linewidth=0.8)

        sns.stripplot(x='Method', y='RankScore', data=sub_rs, hue='InputInference',
                      ax=ax_bar, order=ALL_METHODS, hue_order=INFERENCE_METHODS,
                      dodge=True, color='black', size=5, jitter=0.1, legend=False)
        
        ax_bar.set_xticks(x_base)
        ax_bar.set_xticklabels(ALL_METHODS, rotation=35, ha='right', fontsize=FONT_SIZES['tick_label'])
        ax_bar.set_xlabel("Method", fontsize=FONT_SIZES['axis_title'])
        ax_bar.set_ylabel("Rank-score", fontsize=FONT_SIZES['axis_title'])
        ax_bar.set_title(f"{metric}: Mean Rank-Score ± IQR/2", fontsize=FONT_SIZES['main_title'], fontweight='bold')
        ax_bar.legend(title="Inference method", fontsize=FONT_SIZES['legend_item'], title_fontsize=FONT_SIZES['legend_title'])
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        ax_bar.set_ylim(-0.05, 1.1) 
        ax_bar.text(-0.1, 1.08, panel_labels[col_idx], transform=ax_bar.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')
        ax_bar.tick_params(axis='y', labelsize=FONT_SIZES['tick_label']) 

        # --- Bottom Row: Lollipop Plots ---
        ax_lollipop = axes[1, col_idx]
        sub_friedman = df_friedman[df_friedman['Metric'] == metric].sort_values('AvgRank')
        methods = sub_friedman['Method'].tolist()
        ranks = sub_friedman['AvgRank'].values
        
        ax_lollipop.hlines(y=range(len(methods)), xmin=0, xmax=ranks, color='#2E86AB', linewidth=2.5, alpha=0.8)
        lollipop_colors = ['#A23B72' if i == 0 else '#F18F01' for i in range(len(ranks))]
        ax_lollipop.scatter(ranks, range(len(methods)), color=lollipop_colors, s=120, zorder=3, edgecolors='white', linewidth=1.5)

        for i, r in enumerate(ranks):
            ax_lollipop.text(r + 0.08, i, f"{r:.2f}", va='center', ha='left', fontsize=FONT_SIZES['lollipop_label'], fontweight='bold')

        ax_lollipop.set_yticks(range(len(methods)))
        ax_lollipop.set_yticklabels(methods, fontsize=FONT_SIZES['tick_label'])
        ax_lollipop.set_title(f"{metric}: Average Rank", fontsize=FONT_SIZES['main_title'], fontweight='bold')
        ax_lollipop.set_xlabel('Average Rank (Friedman Test)', fontsize=FONT_SIZES['axis_title']) # Added X-label
        ax_lollipop.set_xlim(0, max(ranks) + 0.8)
        ax_lollipop.grid(axis='x', alpha=0.3, linestyle='--')
        ax_lollipop.invert_yaxis()
        ax_lollipop.text(-0.1, 1.08, panel_labels[2 + col_idx], transform=ax_lollipop.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')
        ax_lollipop.tick_params(axis='x', labelsize=FONT_SIZES['tick_label'])

    plt.tight_layout(pad=3.0)
    filepath = os.path.join(RESULTS_DIR, 'Figure_7') 
    plt.savefig(f"{filepath}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filepath}.eps", format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"Combined rank analysis plot saved to {filepath}.[png/eps]")

def plot_combined_rank_analysis():

    FONT_SIZES = {
        'panel_label': 20, 'main_title': 18, 'axis_title': 16,
        'tick_label': 14, 'legend_title': 16, 'legend_item': 14,
        'lollipop_label': 12, 'stats_text': 12
    }
    inference_labels = {
        'pearson': 'Pearson', 'spearman': 'Spearman',
        'grnboost': 'GRNBoost2', 'genie3': 'GENIE3'
    }

    try:
        df_rs = pd.read_csv(os.path.join(RESULTS_DIR, "dream5_rankscore_details.csv"))
        df_friedman = pd.read_csv(os.path.join(RESULTS_DIR, "friedman_nemenyi_summary.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please run the updated analysis script first.")
        return

    methods_to_plot = [m for m in ALL_METHODS if m != 'Base']
    df_rs = df_rs[df_rs['Method'].isin(methods_to_plot)]
    df_friedman = df_friedman[df_friedman['Method'].isin(methods_to_plot)]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    panel_labels = ['(A)', '(B)', '(C)', '(D)']
    metrics = ['AUROC', 'AUPR']

    for col_idx, metric in enumerate(metrics):

        ax_bar = axes[0, col_idx]
        sub_rs = df_rs[df_rs['Metric'] == metric]
        mean_tbl = sub_rs.pivot_table(index="Method", columns="InputInference", values="RankScore").reindex(index=methods_to_plot, columns=INFERENCE_METHODS)
        err_tbl = sub_rs.pivot_table(index="Method", columns="InputInference", values="RankScore", aggfunc=lambda x: (x.quantile(0.75) - x.quantile(0.25)) / 2).reindex_like(mean_tbl)
        x_base = np.arange(len(methods_to_plot))
        bar_w = 0.18
        colors = ["#4c72b0", "#dd8452", "#c44e52", "#55a868"]
        for j, eng in enumerate(INFERENCE_METHODS):
            xs = x_base + (j - (len(INFERENCE_METHODS) - 1) / 2) * bar_w
            label = inference_labels.get(eng, eng.capitalize())
            ax_bar.bar(xs, mean_tbl[eng], width=bar_w, color=colors[j], edgecolor='k', linewidth=0.4, alpha=0.85, label=label)
            ax_bar.errorbar(xs, mean_tbl[eng], yerr=err_tbl[eng], fmt='none', ecolor='k', capsize=3, linewidth=0.8)
        sns.stripplot(x='Method', y='RankScore', data=sub_rs, hue='InputInference',
                      ax=ax_bar, order=methods_to_plot, hue_order=INFERENCE_METHODS,
                      dodge=True, color='black', size=5, jitter=0.1, legend=False)
        ax_bar.set_xticks(x_base)
        ax_bar.set_xticklabels(methods_to_plot, rotation=35, ha='right', fontsize=FONT_SIZES['tick_label'])
        ax_bar.set_xlabel("Method", fontsize=FONT_SIZES['axis_title'])
        ax_bar.set_ylabel("Rank-score", fontsize=FONT_SIZES['axis_title'])
        ax_bar.set_title(f"{metric}: Mean Rank-Score ± IQR/2", fontsize=FONT_SIZES['main_title'], fontweight='bold')
        ax_bar.legend(title="Inference method", fontsize=FONT_SIZES['legend_item'], title_fontsize=FONT_SIZES['legend_title'])
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        ax_bar.set_ylim(-0.05, 1.1)
        ax_bar.text(-0.1, 1.08, panel_labels[col_idx], transform=ax_bar.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')
        ax_bar.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])

        ax_lollipop = axes[1, col_idx]
        sub_friedman = df_friedman[df_friedman['Metric'] == metric].sort_values('AvgRank')
        methods = sub_friedman['Method'].tolist()
        ranks = sub_friedman['AvgRank'].values
        ax_lollipop.hlines(y=range(len(methods)), xmin=0, xmax=ranks, color='#2E86AB', linewidth=2.5, alpha=0.8)
        lollipop_colors = ['#A23B72' if i == 0 else '#F18F01' for i in range(len(ranks))]
        ax_lollipop.scatter(ranks, range(len(methods)), color=lollipop_colors, s=120, zorder=3, edgecolors='white', linewidth=1.5)
        for i, r in enumerate(ranks):
            ax_lollipop.text(r + 0.08, i, f"{r:.2f}", va='center', ha='left', fontsize=FONT_SIZES['lollipop_label'], fontweight='bold')
        ax_lollipop.set_yticks(range(len(methods)))
        ax_lollipop.set_yticklabels(methods, fontsize=FONT_SIZES['tick_label'])
        ax_lollipop.set_title(f"{metric}: Average Rank", fontsize=FONT_SIZES['main_title'], fontweight='bold')
        ax_lollipop.set_xlabel('Average Rank (Friedman Test)', fontsize=FONT_SIZES['axis_title'])
        ax_lollipop.set_xlim(0, max(ranks) + 1.2) 
        ax_lollipop.grid(axis='x', alpha=0.3, linestyle='--')
        ax_lollipop.invert_yaxis()
        ax_lollipop.text(-0.1, 1.08, panel_labels[2 + col_idx], transform=ax_lollipop.transAxes, fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top', ha='left')
        ax_lollipop.tick_params(axis='x', labelsize=FONT_SIZES['tick_label'])

        if not sub_friedman.empty:
            stats = sub_friedman.iloc[0]
            chi2_val = stats['Friedman_Chi2']
            p_val = stats['Friedman_p']
            stats_text = fr"Friedman Test:" + "\n" + fr"$\chi^2 = {chi2_val:.2f}$, $p = {p_val:.2e}$"
            ax_lollipop.text(0.97, 0.97, stats_text, 
                 transform=ax_lollipop.transAxes,
                 fontsize=FONT_SIZES['stats_text'], 
                 verticalalignment='top', 
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.8, ec='gray'))

    plt.tight_layout(pad=3.0)
    filepath = os.path.join(RESULTS_DIR, 'Figure_7') 
    plt.savefig(f"{filepath}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filepath}.eps", format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"Combined rank analysis plot saved to {filepath}.[png/eps]")

if __name__ == "__main__":
    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory at: {RESULTS_DIR}")
        print("Please ensure result CSVs are present before plotting.")

    plot_facet_heatmap()
    plot_performance_distributions()
    plot_combined_improvement_analysis()
    plot_combined_rank_analysis()
    
    print("\nAll DREAM5 plots have been generated successfully.")