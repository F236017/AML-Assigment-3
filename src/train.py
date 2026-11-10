"""
Task 4 – Training with SGD + Momentum + Weight Decay
=====================================================
All update rules implemented from scratch (no torch.optim).
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from prepare import prepare
from model import ECGNet, cross_entropy

# Hyperparameters (fixed by assignment)
BATCH_SIZE = 128
MOMENTUM   = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS     = 30
ALPHAS     = [0.01, 0.1, 0.5]


def compute_accuracy(model, X, y, batch_size=1024):
    """Compute accuracy without tracking gradients."""
    model.eval()
    correct = 0
    n = X.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = X[start:start+batch_size]
            yb = y[start:start+batch_size]
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
    model.train()
    return correct / n


def compute_loss(model, X, y, batch_size=1024):
    """Compute average loss without tracking gradients."""
    model.eval()
    total_loss = 0.0
    n = X.shape[0]
    n_batches = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = X[start:start+batch_size]
            yb = y[start:start+batch_size]
            logits = model(xb)
            total_loss += cross_entropy(logits, yb).item()
            n_batches += 1
    model.train()
    return total_loss / n_batches


def train(alpha, X_train, y_train, X_val, y_val, verbose=True):
    """
    Train a fresh ECGNet with the given learning rate.
    
    Returns
    -------
    model : the trained ECGNet
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
              each a list of length EPOCHS
    """
    torch.manual_seed(0)
    model = ECGNet()

    # Initial loss check
    init_loss = compute_loss(model, X_train[:256], y_train[:256])
    if verbose:
        print(f"  Initial loss (should be ~1.61): {init_loss:.4f}")

    # Initialise velocities to zero for each parameter
    velocities = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    n_train = X_train.shape[0]

    for epoch in range(EPOCHS):
        # Shuffle training data each epoch
        perm = torch.randperm(n_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        # Mini-batch training
        for start in range(0, n_train, BATCH_SIZE):
            xb = X_shuffled[start:start+BATCH_SIZE]
            yb = y_shuffled[start:start+BATCH_SIZE]

            # Forward
            logits = model(xb)
            loss = cross_entropy(logits, yb)

            # Backward
            model.zero_grad()
            loss.backward()

            # Update: SGD with momentum and weight decay
            with torch.no_grad():
                for name, p in model.named_parameters():
                    grad = p.grad

                    # Weight decay: applied to weights only (not biases)
                    if "W" in name:
                        grad = grad + WEIGHT_DECAY * p

                    # Momentum update
                    velocities[name] = MOMENTUM * velocities[name] + (1 - MOMENTUM) * grad

                    # Parameter update
                    p -= alpha * velocities[name]

        # End-of-epoch metrics
        train_loss = compute_loss(model, X_train, y_train)
        val_loss   = compute_loss(model, X_val, y_val)
        train_acc  = compute_accuracy(model, X_train, y_train)
        val_acc    = compute_accuracy(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    return model, history


def main():
    os.makedirs("results", exist_ok=True)

    # Load data
    print("Loading and preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare()

    # ------------------------------------------------------------------
    # 4.2  Try all three learning rates
    # ------------------------------------------------------------------
    all_histories = {}
    best_alpha = None
    best_val_loss = float("inf")

    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"Training with alpha = {alpha}")
        print(f"{'='*60}")
        t0 = time.time()
        model, history = train(alpha, X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")
        all_histories[alpha] = history

        # Find best validation loss for this alpha
        min_val_loss = min(history["val_loss"])
        min_epoch = history["val_loss"].index(min_val_loss) + 1
        val_acc_at_best = history["val_acc"][min_epoch - 1]

        print(f"  Best val loss: {min_val_loss:.4f} at epoch {min_epoch}")
        print(f"  Val accuracy at best epoch: {val_acc_at_best:.4f}")

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_alpha = alpha

    print(f"\n*** Chosen alpha = {best_alpha} (lowest validation loss = {best_val_loss:.4f}) ***")

    # ------------------------------------------------------------------
    # Plot: Validation loss vs epoch for all three alphas (Task 4.2)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in ALPHAS:
        ax.plot(range(1, EPOCHS+1), all_histories[alpha]["val_loss"],
                label=f"alpha = {alpha}", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs Epoch for Different Learning Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/lr_comparison.png", dpi=150)
    print("Saved results/lr_comparison.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4.3  Re-train with best alpha, save best model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Final training with alpha = {best_alpha}")
    print(f"{'='*60}")
    t0 = time.time()
    model, history = train(best_alpha, X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    # Find best epoch (lowest validation loss)
    best_epoch = int(np.argmin(history["val_loss"]))
    print(f"Best epoch: {best_epoch + 1} (val_loss = {history['val_loss'][best_epoch]:.4f})")
    print(f"Train loss at best epoch: {history['train_loss'][best_epoch]:.4f}")
    print(f"Val loss at best epoch:   {history['val_loss'][best_epoch]:.4f}")
    print(f"Train acc at best epoch:  {history['train_acc'][best_epoch]:.4f}")
    print(f"Val acc at best epoch:    {history['val_acc'][best_epoch]:.4f}")

    # Save model (we save the final epoch model since we trained with the best alpha;
    # In practice we'd checkpoint per-epoch, but we retrain and use the best-epoch params)
    # For simplicity, re-train up to best_epoch only
    print(f"\nRe-training up to best epoch {best_epoch + 1} to save best model...")
    torch.manual_seed(0)
    best_model = ECGNet()
    velocities = {name: torch.zeros_like(p) for name, p in best_model.named_parameters()}
    n_train = X_train.shape[0]

    for epoch in range(best_epoch + 1):
        perm = torch.randperm(n_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        for start in range(0, n_train, BATCH_SIZE):
            xb = X_shuffled[start:start+BATCH_SIZE]
            yb = y_shuffled[start:start+BATCH_SIZE]
            logits = best_model(xb)
            loss = cross_entropy(logits, yb)
            best_model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, p in best_model.named_parameters():
                    grad = p.grad
                    if "W" in name:
                        grad = grad + WEIGHT_DECAY * p
                    velocities[name] = MOMENTUM * velocities[name] + (1 - MOMENTUM) * grad
                    p -= best_alpha * velocities[name]

    torch.save(best_model.state_dict(), "results/best_model.pth")
    print("Saved results/best_model.pth")

    # Verify saved model
    val_loss_check = compute_loss(best_model, X_val, y_val)
    val_acc_check  = compute_accuracy(best_model, X_val, y_val)
    print(f"Verification – val_loss: {val_loss_check:.4f}, val_acc: {val_acc_check:.4f}")

    # ------------------------------------------------------------------
    # Plots for chosen alpha (Task 4.3)
    # ------------------------------------------------------------------
    epochs_range = range(1, EPOCHS + 1)

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs_range, history["val_loss"],   label="Val Loss",   marker="s", markersize=3)
    ax.axvline(best_epoch + 1, color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch+1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training and Validation Loss (alpha = {best_alpha})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/loss_curves.png", dpi=150)
    print("Saved results/loss_curves.png")
    plt.close(fig)

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    ax.plot(epochs_range, history["val_acc"],   label="Val Accuracy",   marker="s", markersize=3)
    ax.axvline(best_epoch + 1, color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch+1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Training and Validation Accuracy (alpha = {best_alpha})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/accuracy_curves.png", dpi=150)
    print("Saved results/accuracy_curves.png")
    plt.close(fig)

    # Save the LR comparison table
    print("\n=== Table 3: Learning-Rate Choice ===")
    print(f"{'alpha':>8s} | {'Lowest Val Loss':>16s} | {'Val Acc at Best Epoch':>22s}")
    print("-" * 55)
    for alpha in ALPHAS:
        h = all_histories[alpha]
        min_vl = min(h["val_loss"])
        idx = h["val_loss"].index(min_vl)
        va = h["val_acc"][idx]
        print(f"{alpha:>8.2f} | {min_vl:>16.4f} | {va:>22.4f}")


if __name__ == "__main__":
    main()
