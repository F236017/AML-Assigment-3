"""
Task 1 – Load and prepare the data
====================================
Steps 1.1 – 1.7 of the assignment.
"""

import numpy as np
import torch
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_raw
from plot_beats import plot_one_per_class

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def prepare():
    """
    Execute all Task-1 steps and return the prepared data.
    
    Returns
    -------
    X_train, y_train : torch tensors (float32, long) – training split
    X_val,   y_val   : torch tensors                  – validation split
    X_test,  y_test  : torch tensors                  – test split
    """

    # ------------------------------------------------------------------
    # 1.1  Load the two CSV files
    # ------------------------------------------------------------------
    raw = load_raw()
    train_df = raw["train"]
    test_df  = raw["test"]
    print("1.1  Shapes loaded")
    print(f"     train_df : {train_df.shape}")
    print(f"     test_df  : {test_df.shape}")

    # ------------------------------------------------------------------
    # 1.2  Separate signal from label; convert label to int
    # ------------------------------------------------------------------
    print(f"\n1.2  Label column dtype (train): {train_df[187].dtype}")
    train_df[187] = train_df[187].astype(int)
    test_df[187]  = test_df[187].astype(int)
    print(f"     After conversion          : {train_df[187].dtype}")

    # ------------------------------------------------------------------
    # 1.3  Report min / max signal value in training file
    # ------------------------------------------------------------------
    signals_train = train_df.iloc[:, :187]
    sig_min = signals_train.values.min()
    sig_max = signals_train.values.max()
    print(f"\n1.3  Signal range (train): min = {sig_min:.6f}, max = {sig_max:.6f}")

    # ------------------------------------------------------------------
    # 1.4  Rows per class
    # ------------------------------------------------------------------
    print("\n1.4  Rows per class")
    for split_name, df in [("train", train_df), ("test", test_df)]:
        counts = df[187].value_counts().sort_index()
        print(f"     {split_name}:")
        for k in range(5):
            print(f"       Class {k} ({CLASS_NAMES[k]}): {counts[k]:>6d}")
        print(f"       Total: {len(df)}")

    # ------------------------------------------------------------------
    # 1.5  Plot one example beat per class
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    plot_one_per_class(raw["train"], out_path="results/example_beats.png")
    print("\n1.5  Example beats plot saved to results/example_beats.png")
    print("     Description: The N (normal) beat shows a clear, tall R-peak with")
    print("     smooth P and T waves. S (supraventricular) beats look similar to N")
    print("     but with a slightly different P-wave morphology and timing. V")
    print("     (ventricular) beats are wider and have an abnormal shape with no")
    print("     clear P-wave. F (fusion) beats are a mix of normal and ventricular")
    print("     morphology. Q (unknown/paced) beats show sharp, narrow spikes")
    print("     characteristic of pacemaker stimulation.")

    # ------------------------------------------------------------------
    # 1.6  Split training file into train and validation
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    val_df   = train_df.iloc[val_idx].reset_index(drop=True)
    train_split_df = train_df.iloc[train_idx].reset_index(drop=True)

    print(f"\n1.6  Split sizes")
    print(f"     train      : {len(train_split_df)}")
    print(f"     validation : {len(val_df)}")
    print(f"     test       : {len(test_df)}")

    # ------------------------------------------------------------------
    # 1.7  Convert to tensors
    # ------------------------------------------------------------------
    X_train = torch.tensor(train_split_df.iloc[:, :187].values, dtype=torch.float32)
    y_train = torch.tensor(train_split_df[187].values,          dtype=torch.long)

    X_val   = torch.tensor(val_df.iloc[:, :187].values,         dtype=torch.float32)
    y_val   = torch.tensor(val_df[187].values,                  dtype=torch.long)

    X_test  = torch.tensor(test_df.iloc[:, :187].values,        dtype=torch.float32)
    y_test  = torch.tensor(test_df[187].values,                 dtype=torch.long)

    print(f"\n1.7  Tensor shapes and dtypes")
    print(f"     X_train: {X_train.shape}, {X_train.dtype}")
    print(f"     y_train: {y_train.shape}, {y_train.dtype}")
    print(f"     X_val  : {X_val.shape},  {X_val.dtype}")
    print(f"     y_val  : {y_val.shape},  {y_val.dtype}")
    print(f"     X_test : {X_test.shape}, {X_test.dtype}")
    print(f"     y_test : {y_test.shape}, {y_test.dtype}")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare()
    print("\nTask 1 complete.")
