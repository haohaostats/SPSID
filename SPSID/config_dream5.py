
# config_dream5.py

import os

# --- Base Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Folder Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data", "dream5")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "dream5")

GOLD_STANDARD_DIR = os.path.join(DATA_DIR, "Gold_Standard")
GRN_NETWORK_DIR = os.path.join(DATA_DIR, "GRN_Network")

# --- File and Network Definitions ---
NETWORK_INFO = {
    1: "DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv",
    2: "DREAM5_NetworkInference_GoldStandard_Network2 - S. aureus.txt",
    3: "DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv",
    4: "DREAM5_NetworkInference_GoldStandard_Network4 - S. cerevisiae.tsv",
}

INFERENCE_METHODS = ["pearson", "spearman", "grnboost", "genie3"]
ALL_METHODS = ["Base", "SPSID", "RENDOR", "ND", "NE", "Silencer", "ICM"]
EPS = 1e-8