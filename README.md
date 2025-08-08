
# SPSID: Single-Parameter Shrinkage Inverse Diffusion for GRN Denoising

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Research%20Code-orange.svg)

**SPSID** is a lightweight inverse-diffusion filter that denoises genome-wide **gene regulatory networks (GRNs)** with a **single shrinkage parameter**.  
This repository provides **reproducible pipelines** for **simulation studies** and the **DREAM5** benchmark, including analysis and publication-quality plots.

> TL;DR: Put data under `SPSID/data/...`, install requirements, run `python SPSID/run_*` scripts, and figures/tables will appear under `SPSID/results/`.

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Data Setup](#data-setup)
- [Reproducible Pipelines](#reproducible-pipelines)
  - [1) Simulation](#1-simulation)
  - [2) DREAM5 Evaluation](#2-dream5-evaluation)
- [Outputs & Where to Find Them](#outputs--where-to-find-them)
- [Configuration Files](#configuration-files)
- [Method at a Glance](#method-at-a-glance)
- [Reproducibility Tips](#reproducibility-tips)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Maintainer](#maintainer)

---

## Highlights

- **Single-parameter** shrinkage inverse diffusion; simple to tune.
- **Plug-and-play**: takes a noisy network and outputs a denoised one.
- **End-to-end scripts** for Simulation and DREAM5, with ready-to-use figures.

---

## Repository Structure

```
SPSID/
├─ run_simulation.py
├─ analyze_simulation_results.py
├─ plot_simulation_results.py
├─ run_dream5_evaluation.py
├─ analyze_dream5_results.py
├─ plot_dream5_results.py
├─ methods.py                # SPSID and baseline methods
├─ utils.py
├─ utils_dream5.py
├─ config_simulation.py      # knobs for the simulation pipeline
├─ config_dream5.py          # expected filenames/paths for DREAM5
├─ data/
│  └─ dream5/
│     ├─ Gold_Standard/      # place DREAM5 gold-standard files here
│     └─ GRN_Network/        # place inferred networks here
└─ results/
   ├─ simulation/            # generated figures & CSVs
   └─ dream5/                # generated figures & CSVs
```

---

## Requirements & Installation

- **Python**: 3.9 or newer
- Works on **Linux / macOS / Windows**

```bash
# 1) (Recommended) create and activate a clean environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

---

## Data Setup

### DREAM5 (required only for the DREAM5 pipeline)

1) Obtain **DREAM5 Network Inference** datasets (gold standards and inferred networks) from **Synapse**  
   (dataset commonly referenced as: *DREAM5 Network Inference Challenge*, e.g., ID `syn2787211`).  
2) Place the files under:
```bash
SPSID/data/dream5/Gold_Standard/   # gold-standard edge lists
SPSID/data/dream5/GRN_Network/     # inferred networks from GRN methods
```
3) Verify the **expected filenames/paths** in `SPSID/config_dream5.py`.  
   Adjust that file if your local filenames differ.

---

## Reproducible Pipelines

### 1) Simulation

```bash
python SPSID/run_simulation.py
python SPSID/analyze_simulation_results.py
python SPSID/plot_simulation_results.py
```

Generates:
- Synthetic networks and noisy observations
- Denoised networks by SPSID and baselines
- AUROC/AUPR, summary stats
- Figures and tables in `SPSID/results/simulation/`

---

### 2) DREAM5 Evaluation

```bash
python SPSID/run_dream5_evaluation.py
python SPSID/analyze_dream5_results.py
python SPSID/plot_dream5_results.py
```

Generates:
- Denoised DREAM5 networks
- AUROC/AUPR, rank-score
- Statistical tests
- Figures and tables in `SPSID/results/dream5/`

---

## Outputs & Where to Find Them

**Simulation** (`SPSID/results/simulation/`):
- `performance_comparison_results.csv`
- `lambda_sensitivity_summary.csv`
- `Figure_2.png`, `Figure_3.png`

**DREAM5** (`SPSID/results/dream5/`):
- `dream5_overall_scores.csv`
- `improvement_overall_median_iqr.csv`
- `friedman_nemenyi_summary.csv`
- Publication-ready plots

---

## Configuration Files

- `SPSID/config_simulation.py` — controls synthetic network size, sparsity, noise, SPSID parameter(s)  
- `SPSID/config_dream5.py` — expected filenames/paths for DREAM5 datasets

---

## Method at a Glance

```mermaid
flowchart LR
  A[Noisy network W_obs] --> B[Preprocess\n(ε-stabilization)]
  B --> C[Row-stochastic P_obs]
  C --> D[Single-parameter\nshrinkage inverse diffusion]
  D --> E[Direct-edge estimate W_dir]
  E --> F[Ranking & Evaluation\n(AUROC/AUPR, rank-score)]
```

---

## Reproducibility Tips

- Fix random seeds for simulation
- Keep DREAM5 data layout as in `config_dream5.py`
- Do not commit large datasets

---

## FAQ / Troubleshooting

**Q1:** File not found for DREAM5 → Check `config_dream5.py` paths  
**Q2:** Missing plots/CSVs → Run scripts in order (run → analyze → plot)

---

## Citation

```
@software{chen2025_spsid,
  author  = {Hao Chen and collaborators},
  title   = {SPSID: Single-Parameter Shrinkage Inverse Diffusion for GRN Denoising},
  year    = {2025},
  version = {v0.1.0},
  url     = {https://github.com/haohaostats/SPSID}
}
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Maintainer

- hao hao ([@haohaostats](https://github.com/haohaostats))

