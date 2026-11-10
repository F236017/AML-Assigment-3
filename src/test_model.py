"""
Task 5 – Test the trained model once
=====================================
Load saved parameters, predict on test set, report all metrics.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from prepare import prepare
from model import ECGNet, cross_entropy
from metrics import accuracy, confusion_matrix, per_class_accuracy

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def test_model():
    """Run the full test evaluation and generate all results."""
    os.makedirs("results", exist_ok=True)

    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare()

    # 5.1  Load the saved model
    print("\nLoading saved model from results/best_model.pth...")
    torch.manual_seed(0)
    model = ECGNet()
    model.load_state_dict(torch.load("results/best_model.pth", weights_only=True))
    model.eval()

    # Predict on test set
    with torch.no_grad():
        logits = model(X_test)            # (21892, 5)
        y_pred = logits.argmax(dim=1)     # predicted class

    # 5.2 & 5.3  Metrics
    test_acc = accuracy(y_test, y_pred)
    C = confusion_matrix(y_test, y_pred, n_classes=5)
    pca = per_class_accuracy(C)
    balanced_acc = sum(pca) / len(pca)

    # Always-N accuracy (Task 6.1)
    always_n_acc = (y_test == 0).float().mean().item()

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test accuracy:       {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Balanced accuracy:   {balanced_acc:.4f}  ({balanced_acc*100:.2f}%)")
    print(f"Always-'N' accuracy: {always_n_acc:.4f}  ({always_n_acc*100:.2f}%)")

    print(f"\nPer-class accuracy:")
    for k in range(5):
        print(f"  {CLASS_NAMES[k]}: {pca[k]:.4f}  ({pca[k]*100:.2f}%)")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    header = "      " + "".join(f"{CLASS_NAMES[k]:>7s}" for k in range(5)) + "  | Row Sum"
    print(header)
    print("  " + "-" * len(header))
    for j in range(5):
        row = f"  {CLASS_NAMES[j]:>3s}  "
        for k in range(5):
            row += f"{C[j,k].item():>7d}"
        row += f"  | {C[j].sum().item():>7d}"
        print(row)
    print(f"  Total: {C.sum().item()}")

    # ------------------------------------------------------------------
    # Plot: Confusion matrix heatmap
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    C_np = C.numpy().astype(float)
    im = ax.imshow(C_np, cmap="Blues")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Test Confusion Matrix")

    # Annotate cells
    for j in range(5):
        for k in range(5):
            val = int(C_np[j, k])
            color = "white" if C_np[j, k] > C_np.max() * 0.5 else "black"
            ax.text(k, j, str(val), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    print("\nSaved results/confusion_matrix.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Table 4 – Test results (save to file)
    # ------------------------------------------------------------------
    with open("results/test_results.txt", "w") as f:
        f.write("Table 4 – Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Metric':<25s} | {'Value':>10s}\n")
        f.write("-" * 40 + "\n")
        for k in range(5):
            f.write(f"Per-class acc ({CLASS_NAMES[k]})       | {pca[k]*100:>9.2f}%\n")
        f.write(f"{'Test accuracy':<25s} | {test_acc*100:>9.2f}%\n")
        f.write(f"{'Balanced accuracy':<25s} | {balanced_acc*100:>9.2f}%\n")
        f.write(f"{'Always-N accuracy':<25s} | {always_n_acc*100:>9.2f}%\n")
        f.write("\n\nConfusion Matrix:\n")
        f.write("      " + "".join(f"{CLASS_NAMES[k]:>7s}" for k in range(5)) + "\n")
        for j in range(5):
            row = f"  {CLASS_NAMES[j]:>3s}  "
            for k in range(5):
                row += f"{C[j,k].item():>7d}"
            f.write(row + "\n")

    print("Saved results/test_results.txt")

    return test_acc, balanced_acc, C, pca


if __name__ == "__main__":
    test_model()
