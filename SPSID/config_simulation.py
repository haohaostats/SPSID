
# config_simulation.py

import os
import numpy as np

# --- Base Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Folder Paths ---
RESULTS_DIR = os.path.join(BASE_DIR, "results", "simulation")

# --- Simulation Parameters ---
NUM_TRIALS = 1000
N_NODES = 100

BASIC_PARAMS = {
    'density': 0.05,
    'beta': 0.1,
    'noise': 0.5
}
SENSITIVITY_FIXED_PARAMS = {
    'density': 0.05,
    'beta': 0.5,
    'noise': 0.5
}
BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
SIGMA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
DENSITY_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09]
LAMBDA_GRID = [10, 100] + list(range(200, 1100, 100)) + [5000, 10000]
EDGE_RANGE = np.arange(100, 1501, 100)

# --- Method and Plotting Configurations ---
METHODS = ['SPSID', 'RENDOR', 'ND', 'NE', 'Silencer', 'ICM']
EPS = 1e-8
LAMBDA_VAL = 1000
EPS1 = 1e-6
EPS2 = 1e-6

COLOR_MAP = {
    "SPSID": "blue", "RENDOR": "red", "ND": "green", 
    "NE": "orange", "Silencer": "purple", "ICM": "brown"
}