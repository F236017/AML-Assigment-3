"""
Task 3.2 – Gradient check
==========================
Hand-compute dJ/dW3 and dJ/db3 and compare with PyTorch autograd.
"""

import torch
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from prepare import prepare
from model import ECGNet, cross_entropy


def gradient_check():
    """
    Run forward + backward on first 64 training rows.
    Compute gradients manually and compare with autograd.
    """
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare()

    # Take first 64 training rows
    x_batch = X_train[:64]
    y_batch = y_train[:64]
    B = x_batch.shape[0]

    # Create model (seed 0)
    torch.manual_seed(0)
    model = ECGNet()

    # Forward pass
    logits = model(x_batch)            # (64, 5)
    loss = cross_entropy(logits, y_batch)

    # Backward (autograd)
    loss.backward()

    # ------- Hand-computed gradients -------
    # Y_hat = softmax probabilities (B, 5)
    Y_hat = torch.softmax(logits.detach(), dim=1)

    # Y = one-hot labels (B, 5)
    Y = torch.zeros(B, 5)
    Y[torch.arange(B), y_batch] = 1.0

    # A2 = second hidden layer activations (B, 64)
    A2 = model._a2.detach()

    # dJ/dW3 = (1/B) * (Y_hat - Y)^T @ A2        shape (5, 64)
    dW3_hand = (1.0 / B) * (Y_hat - Y).t() @ A2

    # dJ/db3 = (1/B) * sum_i (y_hat_i - y_i)      shape (5,)
    db3_hand = (1.0 / B) * (Y_hat - Y).sum(dim=0)

    # ------- Compare -------
    dW3_auto = model.W3.grad
    db3_auto = model.b3.grad

    max_diff_W3 = (dW3_hand - dW3_auto).abs().max().item()
    max_diff_b3 = (db3_hand - db3_auto).abs().max().item()

    print("\n=== Gradient Check (Task 3.2) ===")
    print(f"Max |dJ/dW3_hand - dJ/dW3_auto| = {max_diff_W3:.2e}")
    print(f"Max |dJ/db3_hand - dJ/db3_auto| = {max_diff_b3:.2e}")
    print(f"Both should be < 1e-5: {'PASS' if max_diff_W3 < 1e-5 and max_diff_b3 < 1e-5 else 'FAIL'}")

    return max_diff_W3, max_diff_b3


if __name__ == "__main__":
    gradient_check()
