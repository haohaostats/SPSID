
# analyze_dream5_results.py

import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st

from config_dream5 import RESULTS_DIR, ALL_METHODS

def calculate_rank_scores(df):
    """
    计算Rank-score和Overall Scores 方法。
    """
    print("\n--- Calculating Rank-scores and Overall Scores ---")
    
    methods_to_compare = [m for m in ALL_METHODS if m != 'Base']
    df_filtered = df[df['Method'].isin(methods_to_compare)].copy()

    rank_rows = []
    for (net, inf), grp in df_filtered.groupby(["NetworkID", "InputInference"]):
        for metric in ["AUROC", "AUPR"]:
            ser = grp.set_index("Method")[metric]
            ranks = ser.rank(ascending=False, method="min")
            
            scores = 1.0 - (ranks - 1) / (len(ranks) - 1)
            
            for m, rs in scores.items():
                rank_rows.append({"NetworkID": net, "InputInference": inf,
                                  "Method": m, "Metric": metric, "RankScore": rs})
    df_rs = pd.DataFrame(rank_rows)
    df_rs.to_csv(os.path.join(RESULTS_DIR, "dream5_rankscore_details.csv"), index=False)

    df_piv = (df_rs.pivot_table(index="Method", values="RankScore", aggfunc="mean")
                   .rename(columns={"RankScore": "OverallScore"})
                   .sort_values("OverallScore", ascending=False)
                   .reindex(methods_to_compare))
                   
    df_piv.to_csv(os.path.join(RESULTS_DIR, "dream5_overall_scores.csv"))
    print("Rank-score details and overall scores saved.")

def perform_statistical_tests(df):

    print("\n--- Performing Friedman and Nemenyi Tests ---")
    methods_to_compare = [m for m in ALL_METHODS if m != 'Base']

    df['Dataset'] = df['NetworkID'].astype(str) + "_" + df['InputInference']
    
    df_auc = df.pivot(index='Dataset', columns='Method', values='AUROC')[methods_to_compare].dropna()
    df_aupr = df.pivot(index='Dataset', columns='Method', values='AUPR')[methods_to_compare].dropna()

    M_auc, M_aupr = df_auc.to_numpy(), df_aupr.to_numpy()
    stat_auc, p_auc = st.friedmanchisquare(*M_auc.T)
    stat_aupr, p_aupr = st.friedmanchisquare(*M_aupr.T)

    def nemenyi(mat, alpha=0.05):
        N, k = mat.shape 
        ranks = np.apply_along_axis(st.rankdata, 1, -mat)
        avg_r = ranks.mean(axis=0)
        q_alpha = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031}[k]
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
        diff = np.abs(avg_r[:, None] - avg_r[None, :])
        signif = (diff > cd).astype(int)
        return avg_r, cd, signif

    avg_auc, cd_auc, sig_auc = nemenyi(M_auc)
    avg_aupr, cd_aupr, sig_aupr = nemenyi(M_aupr)

    rows = []
    for i, m in enumerate(methods_to_compare):
        rows.append({
            "Metric": "AUROC", "Method": m, "AvgRank": avg_auc[i], "CD": cd_auc,
            "Friedman_Chi2": stat_auc, "Friedman_p": p_auc
        })
        rows.append({
            "Metric": "AUPR", "Method": m, "AvgRank": avg_aupr[i], "CD": cd_aupr,
            "Friedman_Chi2": stat_aupr, "Friedman_p": p_aupr
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "friedman_nemenyi_summary.csv"), index=False)
    
    sig_dict = {
        "AUROC": pd.DataFrame(sig_auc, index=methods_to_compare, columns=methods_to_compare).to_dict(),
        "AUPR": pd.DataFrame(sig_aupr, index=methods_to_compare, columns=methods_to_compare).to_dict()
    }
    with open(os.path.join(RESULTS_DIR, "nemenyi_significance.json"), "w") as f:
        json.dump(sig_dict, f, indent=2)

    print("\n=== Friedman Test Results ===")
    print(f"AUROC : chi-squared={stat_auc:.2f}, p-value={p_auc:.3g}")
    print(f"AUPR  : chi-squared={stat_aupr:.2f}, p-value={p_aupr:.3g}")
    print("\nStatistical analysis results saved.")

def calculate_improvement_tables(df):
    """
    Calculates and saves tables for percentage improvement over Base, now using
    Median and Interquartile Range (IQR).
    """
    print("\n--- Calculating Percentage Improvement Tables (Median and IQR) ---")
    df["Task"] = df["NetworkID"].astype(str) + "_" + df["InputInference"]
    
    base = (df[df["Method"] == "Base"]
            .loc[:, ["Task", "AUROC", "AUPR"]]
            .rename(columns={"AUROC": "Base_AUROC", "AUPR": "Base_AUPR"}))
            
    df_merged = df.merge(base, on="Task", how="left")
    df_merged["dAUROC_%"] = (df_merged["AUROC"] - df_merged["Base_AUROC"]) / df_merged["Base_AUROC"] * 100
    df_merged["dAUPR_%"]  = (df_merged["AUPR"]  - df_merged["Base_AUPR"])  / df_merged["Base_AUPR"]  * 100

    def median_and_iqr(x):
        median = x.median()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        return pd.Series({"median": median, "iqr": iqr})

    overall = (df_merged[df_merged["Method"] != "Base"]
               .groupby("Method")[["dAUROC_%", "dAUPR_%"]]
               .apply(lambda s: pd.concat([median_and_iqr(s["dAUROC_%"]), median_and_iqr(s["dAUPR_%"])], axis=0)))

    overall.columns = ["dAUROC_median", "dAUROC_IQR", "dAUPR_median", "dAUPR_IQR"]
    overall = overall.sort_values("dAUROC_median", ascending=False).round(2)
    overall.to_csv(os.path.join(RESULTS_DIR, "improvement_overall_median_iqr.csv"))
    print("Overall improvement table (Median, IQR) saved.")

    per_net = (df_merged[df_merged["Method"] != "Base"]
               .groupby(["NetworkID", "Method"])[["dAUROC_%", "dAUPR_%"]]
               .apply(lambda s: pd.concat([median_and_iqr(s["dAUROC_%"]), median_and_iqr(s["dAUPR_%"])], axis=0)))

    per_net.columns = ["dAUROC_median", "dAUROC_IQR", "dAUPR_median", "dAUPR_IQR"]
    per_net = (per_net.unstack(level=0).swaplevel(axis=1).sort_index(axis=1, level=0)).round(2)
    per_net.to_csv(os.path.join(RESULTS_DIR, "improvement_per_network_median_iqr.csv"))
    print("Per-network improvement table (Median, IQR) saved.")
    
    print("\n=== Overall (median, IQR, %) ===")
    print(overall)
    print("\n=== Per-network (median, IQR, %) ===")
    print(per_net)

if __name__ == "__main__":

    input_csv = os.path.join(RESULTS_DIR, "dream5_all_performance_results.csv")
    if not os.path.exists(input_csv):
        print(f"Error: Results file not found at {input_csv}")
        print("Please run 'run_dream5_evaluation.py' first.")
    else:
        df_perf = pd.read_csv(input_csv)
    
        # Run all analysis components
        calculate_rank_scores(df_perf.copy())
        perform_statistical_tests(df_perf.copy())
        calculate_improvement_tables(df_perf.copy())
    
        print("\nAll analysis tasks are complete.")