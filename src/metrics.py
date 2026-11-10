"""
Task 5.2 – Metrics
==================
Three metric functions: accuracy, confusion_matrix, per_class_accuracy.
All implemented from scratch (no scikit-learn).
"""

import torch


def accuracy(y_true, y_pred):
    """
    Overall accuracy: fraction of correct predictions.
    
    Parameters
    ----------
    y_true : tensor of shape (m,) – true labels
    y_pred : tensor of shape (m,) – predicted labels
    
    Returns
    -------
    acc : float
    """
    return (y_true == y_pred).float().mean().item()


def confusion_matrix(y_true, y_pred, n_classes=5):
    """
    Confusion matrix C where C[j, k] = number of samples with
    true class j and predicted class k.
    
    Parameters
    ----------
    y_true : tensor of shape (m,)
    y_pred : tensor of shape (m,)
    n_classes : int
    
    Returns
    -------
    C : tensor of shape (n_classes, n_classes)
    """
    C = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for j in range(n_classes):
        mask_j = (y_true == j)
        for k in range(n_classes):
            C[j, k] = ((y_true == j) & (y_pred == k)).sum().item()
    return C


def per_class_accuracy(C):
    """
    Per-class accuracy: for class j, C[j,j] / sum_k C[j,k].
    
    Parameters
    ----------
    C : tensor of shape (n_classes, n_classes) – confusion matrix
    
    Returns
    -------
    pca : list of floats, one per class
    """
    n_classes = C.shape[0]
    pca = []
    for j in range(n_classes):
        row_sum = C[j].sum().item()
        if row_sum > 0:
            pca.append(C[j, j].item() / row_sum)
        else:
            pca.append(0.0)
    return pca


if __name__ == "__main__":
    # Quick test
    y_true = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])
    y_pred = torch.tensor([0, 0, 1, 1, 0, 2, 2, 3, 4, 0])

    print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
    C = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{C}")
    pca = per_class_accuracy(C)
    print(f"Per-class accuracy: {pca}")
    print(f"Balanced accuracy: {sum(pca)/len(pca):.4f}")
