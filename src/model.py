"""
Task 2 & 3.1 – ECGNet and cross-entropy loss
=============================================
All layers are built from nn.Parameter (no nn.Linear, nn.Sequential, etc.).
"""

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    """
    Fully-connected neural network for ECG heartbeat classification.
    
    Architecture
    ------------
    Layer 1 (hidden): 187 -> 128, ReLU
    Layer 2 (hidden): 128 ->  64, ReLU
    Layer 3 (output):  64 ->   5, no activation (logits)
    
    He initialisation for weights, zero initialisation for biases.
    """

    def __init__(self):
        super().__init__()

        # Layer sizes
        sizes = [187, 128, 64, 5]

        # Create weight matrices and bias vectors as nn.Parameter
        # W[l] has shape (n[l], n[l-1])
        # b[l] has shape (n[l],)
        self.W1 = nn.Parameter(torch.randn(sizes[1], sizes[0]) * (2.0 / sizes[0]) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(sizes[1]))

        self.W2 = nn.Parameter(torch.randn(sizes[2], sizes[1]) * (2.0 / sizes[1]) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(sizes[2]))

        self.W3 = nn.Parameter(torch.randn(sizes[3], sizes[2]) * (2.0 / sizes[2]) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(sizes[3]))

    def forward(self, x):
        """
        Forward pass for a mini-batch.
        
        Parameters
        ----------
        x : tensor of shape (B, 187)
        
        Returns
        -------
        logits : tensor of shape (B, 5)
        
        Also stores self._a2 for the gradient check (activation of layer 2).
        """
        # Layer 1: z1 = x @ W1^T + b1,  a1 = ReLU(z1)
        z1 = x @ self.W1.t() + self.b1
        a1 = torch.clamp(z1, min=0)       # ReLU

        # Layer 2: z2 = a1 @ W2^T + b2,  a2 = ReLU(z2)
        z2 = a1 @ self.W2.t() + self.b2
        a2 = torch.clamp(z2, min=0)       # ReLU

        # Store a2 for the gradient check
        self._a2 = a2

        # Layer 3: z3 = a2 @ W3^T + b3  (logits, no activation)
        z3 = a2 @ self.W3.t() + self.b3

        return z3


def cross_entropy(logits, y):
    """
    Softmax cross-entropy loss (numerically stable).
    
    L = -z_y + log(sum(exp(z)))   (per sample)
    
    Uses torch.logsumexp to avoid overflow.
    
    Parameters
    ----------
    logits : tensor of shape (B, C) – raw output of the network
    y      : tensor of shape (B,)   – integer class labels
    
    Returns
    -------
    loss : scalar tensor – mean cross-entropy over the batch
    """
    B = logits.shape[0]
    # Gather the logit for the true class: z_y for each sample
    z_y = logits[torch.arange(B), y]
    # log-sum-exp over classes (dim=1)
    lse = torch.logsumexp(logits, dim=1)
    # Per-sample loss
    losses = -z_y + lse
    return losses.mean()


if __name__ == "__main__":
    # --- 2.4  Print total trainable parameters ---
    torch.manual_seed(0)
    model = ECGNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")
    print()
    print("Calculation by hand:")
    print(f"  W1: 128 x 187 = {128*187}")
    print(f"  b1: 128        = {128}")
    print(f"  W2:  64 x 128 = {64*128}")
    print(f"  b2:  64        = {64}")
    print(f"  W3:   5 x  64 = {5*64}")
    print(f"  b3:   5        = {5}")
    total_hand = 128*187 + 128 + 64*128 + 64 + 5*64 + 5
    print(f"  Total          = {total_hand}")

    # Quick test forward pass
    x_dummy = torch.randn(4, 187)
    logits = model(x_dummy)
    print(f"\nForward pass test – logits shape: {logits.shape}")

    # Test initial loss (should be close to log(5) ≈ 1.6094)
    y_dummy = torch.tensor([0, 1, 2, 3])
    loss = cross_entropy(logits, y_dummy)
    print(f"Initial loss (should be ~1.61): {loss.item():.4f}")
