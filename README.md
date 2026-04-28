## 📁 Code Modules Overview
The technical components of the project are as follows:
- **`model.py` (Architecture Core):** Implements cross-attention blocks based on **Adaptive Gating**, **Tabular AE**, and flow matching logic integrated with **mini-batch OT** (Optimal Transport).
- **`train_handler.py` (Data Preprocessing):** Handles adaptive dimension alignment and inverse density sampling weight calculation tailored to the heterogeneous attributes and long-tail characteristics of traffic data.
- **`train.py` (Joint Training):** Invokes an LLM to extract protocol semantic priors and executes a two-stage training task, spanning from latent space pre-training to flow trajectory optimization.
- **`process.py` (Controlled Generation):** Encapsulates the SDE solver and CFG (Classifier-Free Guidance) logic, supporting the on-demand generation of attack samples for data augmentation.
- **`analyze.py` (Quality Evaluation):** Provides multi-dimensional statistical distribution metrics (KS, Wasserstein) and feature correlation drift analysis.
    

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install torch sentence-transformers pot pandas numpy scikit-learn joblib
```

### 2. Model Training
Configure the dataset path in `train.py` and start the SemflowNet training:
```bash
python train.py
```
_The training process automatically completes semantic extraction -> AE latent space construction -> flow matching trajectory straightening._

### 3. High-Fidelity Traffic Generation
Run `process.py` to generate a synthetic dataset for enhancing the robustness of IDS (Intrusion Detection System) models:
```bash
python process.py
```

### 4. Evaluate Generation Quality
Compare the distribution consistency between the synthetic data and the real data:
```bash
python analyze.py
```
