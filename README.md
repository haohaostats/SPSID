# SPSID: Single-Parameter Shrinkage Inverse Diffusion for GRN Denoising

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Research%20Code-orange.svg)

**SPSID** is a novel network denoising framework designed to accurately infer gene regulatory networks (GRNs) by filtering structural noise from transitive correlations. This repository contains the official implementation of the method described in our manuscript.

---

## 🚀 Quick Reproduction Guide

> **Note:** Data has already been placed under:
> ```
> SPSID/data/dream5/Gold_Standard/
> SPSID/data/dream5/GRN_Network/
> ```
> You can run the pipelines directly.

### 1. Simulation Pipeline
```bash
python SPSID/run_simulation.py
python SPSID/analyze_simulation_results.py
python SPSID/plot_simulation_results.py
```
**Outputs:** `SPSID/results/simulation/` — CSV tables, sensitivity analyses, and publication-ready plots.

### 2. DREAM5 Evaluation Pipeline
```bash
python SPSID/run_dream5_evaluation.py
python SPSID/analyze_dream5_results.py
python SPSID/plot_dream5_results.py
```
**Outputs:** `SPSID/results/dream5/` — AUROC/AUPR scores, rank-scores, statistical test results, improvement summaries, and publication-ready plots.

---

## 📚 Full Guide

### Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Reproducible Pipelines](#reproducible-pipelines)
- [Outputs & Where to Find Them](#outputs--where-to-find-them)
- [Method at a Glance](#method-at-a-glance)
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
├─ methods.py
├─ utils.py
├─ utils_dream5.py
├─ config_simulation.py
├─ config_dream5.py
├─ data/
│  └─ dream5/
│     ├─ Gold_Standard/
│     └─ GRN_Network/
└─ results/
   ├─ simulation/
   └─ dream5/
```

---

## Requirements & Installation

- **Python**: 3.9 or newer  
- Works on **Linux / macOS / Windows**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Reproducible Pipelines

### Simulation
```bash
python SPSID/run_simulation.py
python SPSID/analyze_simulation_results.py
python SPSID/plot_simulation_results.py
```

### DREAM5
```bash
python SPSID/run_dream5_evaluation.py
python SPSID/analyze_dream5_results.py
python SPSID/plot_dream5_results.py
```

---

## Outputs & Where to Find Them

**Simulation (`SPSID/results/simulation/`):**
- `performance_comparison_results.csv`
- `lambda_sensitivity_summary.csv`
- `Figure_2.png`, `Figure_3.png`

**DREAM5 (`SPSID/results/dream5/`):**
- `dream5_overall_scores.csv`
- `improvement_overall_median_iqr.csv`
- `friedman_nemenyi_summary.csv`
- Publication-ready plots

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

## License
MIT License — see [LICENSE](LICENSE)

---

## Maintainer
- hao hao ([@haohaostats](https://github.com/haohaostats))

