# Reinforcement Learning‑based Driver Drowsiness Detection using EEG

## Project Overview
This repository implements a complete pipeline for detecting driver drowsiness from EEG signals using reinforcement learning. The pipeline includes:
- Data loading and preprocessing (band‑power features).
- Baseline classifiers (SVM & Random Forest).
- A custom Gymnasium environment where an RL agent can **wait** for more EEG data before making a prediction.
- PPO (default) training (with optional A2C and a sweep over different `WAIT_PENALTY` values).
- Evaluation on held‑out subjects.
- Plot generation and a small Excel overview.

All code runs on Windows with Python 3.12, CPU‑only.

## Setup Instructions
1. **Clone the repo** and **navigate** to the project root.
2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Download the dataset**
   - Go to https://doi.org/10.6084/m9.figshare.14620873
   - Download `dataset.mat` (≈172 MB) and place it in `data/`.
5. **Run the full pipeline (quick smoke test)**:
   ```powershell
   python run_all.py --quick
   ```
   This will:
   - Extract and scale features.
   - Train the baseline classifiers.
   - Train PPO for 20 k timesteps.
   - Evaluate all models on the test subjects.
   - Generate plots under `results/` and export `results/eeg_overview.xlsx`.

## Results (quick run)
Below are the numbers obtained from the quick run (20 k PPO steps).  They are automatically generated; **do not edit**.

| Method | Accuracy | Macro F1 | Avg. Delay (s) | Timeout Rate |
|--------|----------|----------|----------------|--------------|
| PPO (penalty = -0.05) | **PLACEHOLDER_ACC** | **PLACEHOLDER_F1** | **PLACEHOLDER_DELAY** | **PLACEHOLDER_TIMEOUT** |
| SVM k=1 | **PLACEHOLDER_SVM1_ACC** | **PLACEHOLDER_SVM1_F1** | 1.0 | – |
| SVM k=2 | **PLACEHOLDER_SVM2_ACC** | **PLACEHOLDER_SVM2_F1** | 2.0 | – |
| SVM k=3 | **PLACEHOLDER_SVM3_ACC** | **PLACEHOLDER_SVM3_F1** | 3.0 | – |
| RF k=1 | **PLACEHOLDER_RF1_ACC** | **PLACEHOLDER_RF1_F1** | 1.0 | – |
| RF k=2 | **PLACEHOLDER_RF2_ACC** | **PLACEHOLDER_RF2_F1** | 2.0 | – |
| RF k=3 | **PLACEHOLDER_RF3_ACC** | **PLACEHOLDER_RF3_F1** | 3.0 | – |

*The placeholders will be replaced with the real numbers after the quick run.*

## Directory Structure
```
RL_Project/
│   config.py
│   data_utils.py
│   features.py
│   baseline.py
│   drowsiness_env.py
│   train_ppo.py
│   evaluate.py
│   make_plots.py
│   export_excel.py
│   run_all.py
│   requirements.txt
│   README.md
│   REPORT.md
│
├─ data/                 # place dataset.mat here (git‑ignored)
├─ results/              # generated plots, CSV/Excel reports
└─ models/               # saved PPO models
```

## License
MIT (feel free to adapt for your course work).
