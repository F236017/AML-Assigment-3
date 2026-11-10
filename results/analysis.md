# Analysis – Task 6

## 6.1 Always-"N" Accuracy

A model that always predicts class N (normal beat) would achieve a test accuracy of:

**Always-N accuracy = 18,118 / 21,892 = 82.78%**

Our neural network achieves approximately 97% test accuracy. While 82.78% sounds decent, a model achieving it is completely useless — it never detects any arrhythmia. This shows that **overall accuracy is misleading when classes are heavily imbalanced**. A seemingly high accuracy can simply mean the model learned to predict the majority class. This is exactly why we need metrics like per-class accuracy and balanced accuracy.

## 6.2 Lowest Per-Class Accuracy

The **F (fusion) class** has the lowest per-class accuracy. Looking at the confusion matrix, F beats are most often misclassified as **N (normal)** or **V (ventricular)**. This makes biological sense: fusion beats are, by definition, a blend of a normal sinus beat and a ventricular ectopic beat. Their waveform morphology lies between these two classes, making it inherently ambiguous for a classifier. Additionally, F has only 641 training examples (0.7% of the data), giving the network far fewer opportunities to learn its distinctive features.

## 6.3 Overfitting, Underfitting, or Neither?

Looking at the loss curves, there is a **small but visible gap** between training and validation loss, with training loss being slightly lower. This gap grows mildly as epochs progress, which is a sign of **mild overfitting**. However, the validation loss continues to decrease or plateaus rather than increasing sharply, so the overfitting is not severe. The validation accuracy also remains close to the training accuracy. This suggests the network is well-sized for the problem — large enough to learn meaningful patterns but not so large that it memorises the training data completely.

## 6.4 Why Not Initialise Weights to Zero? Why He Initialisation?

**Initialising all weights to zero** would break the network completely. Every neuron in a layer would compute the same output (since all weights are identical), receive the same gradient, and update identically. This **symmetry** is never broken, so all neurons in a layer remain copies of each other — effectively reducing the layer to a single neuron.

The **He initialisation** uses standard deviation √(2/n[ℓ-1]) specifically for ReLU layers. ReLU sets half its inputs to zero on average, which halves the variance of the output. The factor of 2 in √(2/n) compensates for this halving, keeping the variance of activations roughly constant across layers. Without this, deep networks suffer from vanishing (too-small init) or exploding (too-large init) activations, making training very slow or unstable.

## 6.5 Momentum and Weight Decay

**Momentum** maintains a running average of past gradients. When two consecutive mini-batch gradients point in **opposite directions**, the momentum term partially cancels them out — the velocity vector shrinks, effectively slowing down the update in that oscillating direction. This acts as a damper on noisy, inconsistent gradient directions while reinforcing consistent directions, leading to smoother and faster convergence.

**Weight decay** adds a fraction of the weight's current value to the gradient (g = ∂J/∂P + λP). A weight that receives **no gradient** (zero loss gradient) would still be pushed toward zero at each step (P ← P − α·(λP)), exponentially decaying. This regularises the network by penalising large weights that aren't contributing to reducing the loss.

## 6.6 Helping the Network on Rare Classes

One effective approach would be **class-weighted loss (cost-sensitive learning)**. Instead of the standard cross-entropy, weight each sample's loss by the inverse frequency of its class:

    w_j = total_samples / (n_classes × count_j)

This makes a misclassified F beat (641 samples) contribute ~113× more to the loss than a misclassified N beat (72,471 samples). The network would pay a much higher penalty for getting rare-class predictions wrong, forcing it to learn better decision boundaries for F and S beats.

The idea is straightforward: the standard loss treats every sample equally, so the network naturally focuses on the majority class (where it can reduce the most total loss). By upweighting rare classes, we rebalance the optimisation objective so that every class contributes equally to the total loss, regardless of how many samples it has.
