# ECG Heartbeat Classification – Assignment 3

**Course:** Machine Learning (Fall 2026)  
**Team:** Hammad & Faizan  

## Overview

A fully-connected neural network built **from scratch** in PyTorch to classify ECG heartbeats into 5 categories (N, S, V, F, Q). No pre-built layers (`nn.Linear`, `nn.Sequential`, `nn.ReLU`) or optimisers (`torch.optim`) were used — everything is implemented manually using `nn.Parameter`, `nn.Module`, and `torch` tensors.

## Dataset

- **Source:** [ECG Heartbeat Categorization Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- 87,554 training rows + 21,892 test rows
- 187 signal samples per heartbeat + 1 class label (5 classes)

## Repository Structure

```
data/                    CSV files (not committed)
src/load_data.py         Data loader (provided)
src/plot_beats.py        Plotting helper (provided)
src/prepare.py           Task 1: data preparation
src/model.py             Task 2 & 3.1: ECGNet + cross-entropy
src/grad_check.py        Task 3.2: gradient check
src/train.py             Task 4: training loop
src/metrics.py           Task 5.2: accuracy, confusion matrix, per-class accuracy
src/test_model.py        Task 5: test evaluation
results/                 Figures, tables, analysis, saved model
app/app.py               Task 7: Streamlit front end
README.md                This file
```

## How to Run

### 1. Install dependencies

```bash
pip install torch numpy pandas matplotlib streamlit
```

### 2. Download data

Download `mitbih_train.csv` and `mitbih_test.csv` from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) into the `data/` folder.

### 3. Run all tasks in order

```bash
# Task 1: Load and prepare data
python src/prepare.py

# Task 2: Verify model architecture
python src/model.py

# Task 3.2: Gradient check
python src/grad_check.py

# Task 4: Train the network (tries 3 learning rates, saves best model)
python src/train.py

# Task 5: Test the saved model
python src/test_model.py
```

### 4. Launch the front end (Task 7)

```bash
streamlit run app/app.py
```

## Results Summary

| Metric | Value |
|--------|-------|
| Total parameters | 32,645 |
| Architecture | 187 → 128 (ReLU) → 64 (ReLU) → 5 |
| Test accuracy | ~97% |
| Balanced accuracy | ~85% |
| Always-N baseline | 82.78% |

See `results/analysis.md` for the full analysis (Task 6).
