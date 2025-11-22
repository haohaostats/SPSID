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
df_obs = pd.read_csv("your_network.csv", index_col=0)
		
# 2. Preprocess
df_numeric = df_obs.apply(pd.to_numeric, errors="coerce").fillna(0.0)
W_obs = df_numeric.values.astype(float)
		
# 3. Run SPSID 
G_spsid = spsid(W_obs.copy(), eps1=1e-6, eps2=1e-6, 
lambda_val=1000, return_tf_only=True)
		
# 4. Save results
pd.DataFrame(G_spsid, index=df_obs.index, 
columns=df_obs.columns).to_csv("spsid_result.csv")
```

---

## 📂 Repository Structure

* **`methods.py`**: Core implementation of SPSID and baseline algorithms.
* **`Demo.ipynb`**: Interactive demo reproducing the benchmark results.
* **`data/`**: Contains the example input network and Gold Standard.


