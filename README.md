# SPSID

**Single-Parameter Shrinkage Inverse-Diffusion for Gene Regulatory Network Denoising**

SPSID is a robust, parameter-free method designed to remove structural noise (transitive correlations) from biological networks.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Run the Demo
To reproduce the results above using the included DREAM5 dataset:
```bash
jupyter notebook Demo.ipynb
```
*(Open the notebook and click **"Run All"**)*

---

## 🛠 Usage

You can apply SPSID to your own network (CSV file) with just a few lines of code:

```python
import pandas as pd
from methods import spsid

# 1. Load your network (Rows=TFs, Cols=Targets)
df = pd.read_csv("your_network.csv", index_col=0)

# 2. Run SPSID (default lambda=1000)
denoised_matrix = spsid(df.values, lambda_val=1000)

# 3. Save results
pd.DataFrame(denoised_matrix, index=df.index, columns=df.columns).to_csv("spsid_result.csv")
```

---

## 📂 Repository Structure

* **`methods.py`**: Core implementation of SPSID and baseline algorithms.
* **`Demo.ipynb`**: Interactive demo reproducing the benchmark results.
* **`data/`**: Contains the example input network and Gold Standard.


