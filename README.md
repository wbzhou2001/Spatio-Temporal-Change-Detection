# Spatio-Temporal Change Detection

[![Paper](https://img.shields.io/badge/arXiv-2602.04798-b31b1b.svg)](https://arxiv.org/abs/2602.04798)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#installation)

Score-based sequential **change-point detection** and **region localization**
for **spatio-temporal point processes**.

This repository contains the reference implementation for:

> **Score-Based Change-Point Detection and Region Localization for Spatio-Temporal Point Processes**  
> Wenbin Zhou, Liyan Xie, Shixiang Zhu (2026)  
> https://arxiv.org/abs/2602.04798

**Abstract:**
We study sequential change-point detection for spatio-temporal point processes, where actionable detection requires not only identifying when a distributional change occurs but also localizing where it manifests in space. While classical quickest change detection methods provide strong guarantees on detection delay and false-alarm rates, existing approaches for point-process data predominantly focus on temporal changes and do not explicitly infer affected spatial regions. We propose a likelihood-free, score-based detection framework that jointly estimates the change time and the change region in continuous space-time without assuming parametric knowledge of the pre- or post-change dynamics. The method leverages a localized and conditionally weighted Hyvärinen score to quantify event-level deviations from nominal behavior and aggregates these scores using a spatio-temporal CUSUM-type statistic over a prescribed class of spatial regions. Operating sequentially, the procedure outputs both a stopping time and an estimated change region, enabling real-time detection with spatial interpretability. We establish theoretical guarantees on false-alarm control, detection delay, and spatial localization accuracy, and demonstrate the effectiveness of the proposed approach through simulations and real-world spatio-temporal event data.

---

## ✨ Highlights

- Likelihood-free detection using **score models**
- Sequential CUSUM-style monitoring with false-alarm control
- Continuous-space region localization (not limited to coarse grids)
- Fully reproducible demo notebook included

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/wbzhou2001/Spatio-Temporal-Change-Detection.git
cd Spatio-Temporal-Change-Detection
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install -U pip
pip install numpy scipy torch tqdm matplotlib
```

If you later add a `requirements.txt`, simply use:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1️⃣ Run the Demo Notebook

Open:

```
demo.ipynb
```

Run all cells to reproduce an end-to-end example.

---

### 2️⃣ Minimal Python Example

```python
import numpy as np
from model.detect import SpatioTemporalDetector

# ------------------------------------------------
# 1. Load event data
# ------------------------------------------------
# Expected format: numpy array of shape (N, D)
# Example columns: (t, x, y)
data = np.load("path/to/events.npy")

# ------------------------------------------------
# 2. Define score models
# ------------------------------------------------
# Replace the placeholders below with actual score models
f0 = ...   # score model for nominal regime
f1 = ...   # score model for post-change regime

# ------------------------------------------------
# 3. Initialize detector
# ------------------------------------------------
detector = SpatioTemporalDetector(f0, f1)

# ------------------------------------------------
# 4. Run offline detection
# ------------------------------------------------
results = detector.offline(
    data=data,
    ntt=50,       # number of evaluation time steps
    K=5,          # region search parameter
    nres=20,      # spatial resolution
    radius=0.1,   # localization radius
    use_grid=False,
    verbose=True,
)

print(results)
```

---

## 🧾 Data Format

The repository assumes **spatio-temporal point process data**.

Typical format:

- `data.shape = (N, D)`
- First column: time `t`
- Remaining columns: spatial coordinates (e.g., `x, y`)

Example:

```
[t, x, y]
```

If marks/covariates exist, append them as additional columns.

---

## 📂 Project Structure

```
.
├── cache/            # cached intermediate results
├── model/            # score models + detection logic
├── utils/            # helper utilities
├── demo.ipynb        # demonstration notebook
└── README.md
```

---

## 📖 Citation

If you use this repository, please cite:

```bibtex
@article{zhou2026score,
  title={Score-Based Change-Point Detection and Region Localization for Spatio-Temporal Point Processes},
  author={Zhou, Wenbin and Xie, Liyan and Zhu, Shixiang},
  journal={arXiv preprint arXiv:2602.04798},
  year={2026}
}
```

## 📬 Contact

Wenbin Zhou  

For questions or bug reports, please open a GitHub issue.
