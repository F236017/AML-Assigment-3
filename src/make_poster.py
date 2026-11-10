"""
Generate results/poster.png  —  A4-landscape academic poster
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from load_data import load_raw

# ── config ──────────────────────────────────────────────────────────────────
OUT   = os.path.join(os.path.dirname(__file__), "..", "results", "poster.png")
DATA  = os.path.join(os.path.dirname(__file__), "..", "data")
RES   = os.path.join(os.path.dirname(__file__), "..", "results")

NAVY   = "#0f3460"
TEAL   = "#16213e"
ACCENT = "#e94560"
LIGHT  = "#f0f4f8"
WHITE  = "#ffffff"
GRAY   = "#7f8c8d"

CLASS_NAMES  = ["N\nNormal", "S\nSupraventr.", "V\nVentricular", "F\nFusion", "Q\nUnknown"]
CLASS_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
FS = 125

# ── load data ───────────────────────────────────────────────────────────────
print("Loading data for poster…")
raw      = load_raw(DATA)
train_df = raw["train"]

# ── figure layout ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(23.4, 16.5))   # A1 landscape in inches @ 100dpi
fig.patch.set_facecolor(NAVY)

# outer margin rect
outer = fig.add_axes([0, 0, 1, 1])
outer.set_axis_off()
outer.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                               boxstyle="round,pad=0.005",
                               facecolor=LIGHT, edgecolor=NAVY, linewidth=4,
                               transform=fig.transFigure, zorder=0))

# ── HEADER ──────────────────────────────────────────────────────────────────
header_ax = fig.add_axes([0.01, 0.885, 0.98, 0.1])
header_ax.set_axis_off()
header_ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                   boxstyle="round,pad=0.02",
                                   facecolor=NAVY, edgecolor="none",
                                   transform=header_ax.transAxes))
header_ax.text(0.5, 0.65, "ECG Heartbeat Classification with a Neural Network Built from Scratch",
               transform=header_ax.transAxes, ha="center", va="center",
               fontsize=28, fontweight="bold", color=WHITE, fontfamily="DejaVu Sans")
header_ax.text(0.5, 0.2,
               "Machine Learning Assignment 3  ·  Hammad & Faizan  ·  Fall 2026   |   "
               "Dataset: MIT-BIH Arrhythmia Database (Kaggle)  ·  87,554 train / 21,892 test samples",
               transform=header_ax.transAxes, ha="center", va="center",
               fontsize=13, color="#aed6f1")

# ── grid: 3 columns ─────────────────────────────────────────────────────────
gs = gridspec.GridSpec(
    4, 3,
    left=0.025, right=0.975,
    top=0.875, bottom=0.03,
    wspace=0.055, hspace=0.38,
)


def card_ax(fig, spec, title, title_color=NAVY):
    ax = fig.add_subplot(spec)
    ax.set_facecolor(WHITE)
    for sp in ax.spines.values():
        sp.set_color("#dee2e6")
        sp.set_linewidth(1.2)
    ax.tick_params(labelsize=10, colors="#2c3e50")
    # title band
    ax.set_title(title, fontsize=13, fontweight="bold", color=WHITE,
                 pad=8, loc="center",
                 bbox=dict(facecolor=title_color, edgecolor="none",
                           pad=4.5, boxstyle="round,pad=0.3"))
    return ax


# ══════════════════════════════════════════════════════════
# COLUMN 1: Problem & Data
# ══════════════════════════════════════════════════════════

# [0,0] text: problem statement
ax_prob = fig.add_subplot(gs[0, 0])
ax_prob.set_facecolor(WHITE)
ax_prob.set_axis_off()
ax_prob.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                  boxstyle="round,pad=0.02",
                                  facecolor=WHITE, edgecolor="#dee2e6",
                                  linewidth=1.5, transform=ax_prob.transAxes))
ax_prob.text(0.5, 0.95, "The Problem", transform=ax_prob.transAxes,
             ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)
ax_prob.text(0.07, 0.78,
             "• Input: one heartbeat – 187 ECG samples (125 Hz)\n"
             "• Output: one of 5 classes (N, S, V, F, Q)\n"
             "• No pre-built layers – weights as nn.Parameter\n"
             "• No torch.optim – update rule written by hand\n"
             "• Loss: softmax cross-entropy via logsumexp",
             transform=ax_prob.transAxes, ha="left", va="top",
             fontsize=11.5, color="#2c3e50", linespacing=1.75)

ax_prob.text(0.5, 0.33,
             "\"How well does a small neural network\n"
             " recognise each type of heartbeat?\"",
             transform=ax_prob.transAxes, ha="center", va="top",
             fontsize=12, color=ACCENT, style="italic",
             bbox=dict(facecolor="#fdecea", edgecolor=ACCENT,
                       boxstyle="round,pad=0.4", linewidth=1.5))
ax_prob.text(0.5, 0.0, "Architecture:  187 → 128 (ReLU) → 64 (ReLU) → 5     |     32,645 parameters",
             transform=ax_prob.transAxes, ha="center", va="bottom",
             fontsize=11, color=NAVY, fontweight="bold")

# [1,0] class distribution (train)
ax_dist = card_ax(fig, gs[1, 0], "Class Distribution — Training Set", NAVY)
counts = [72471, 2223, 5788, 641, 6431]
bars = ax_dist.bar(CLASS_NAMES, counts, color=CLASS_COLORS, width=0.6, edgecolor="white", linewidth=1.2)
ax_dist.set_ylabel("Samples", fontsize=11)
ax_dist.set_yscale("log")
ax_dist.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, cnt in zip(bars, counts):
    ax_dist.text(bar.get_x() + bar.get_width()/2, cnt * 1.3,
                 f"{cnt:,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax_dist.spines[["top","right"]].set_visible(False)
ax_dist.set_facecolor(WHITE)

# [2,0] example beats
ax_beats = card_ax(fig, gs[2:, 0], "Example Beats — One Per Class", NAVY)
ax_beats.set_axis_off()
# embed existing figure
img_path = os.path.join(RES, "example_beats.png")
if os.path.exists(img_path):
    img = plt.imread(img_path)
    ax_beats.imshow(img, aspect="auto", extent=[0,1,0,1],
                    transform=ax_beats.transAxes)
else:
    ax_beats.text(0.5, 0.5, "Run prepare.py first", ha="center", va="center")

# ══════════════════════════════════════════════════════════
# COLUMN 2: Training & Architecture
# ══════════════════════════════════════════════════════════

# [0,1] architecture diagram
ax_arch = fig.add_subplot(gs[0, 1])
ax_arch.set_facecolor(WHITE)
ax_arch.set_axis_off()
ax_arch.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                  boxstyle="round,pad=0.02",
                                  facecolor=WHITE, edgecolor="#dee2e6",
                                  linewidth=1.5, transform=ax_arch.transAxes))
ax_arch.text(0.5, 0.95, "Network Architecture & Training", transform=ax_arch.transAxes,
             ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)

# draw layer boxes
layers = [("Input\n187", 0.08, "#ecf0f1", "#2c3e50"),
          ("Hidden 1\n128  ReLU", 0.32, "#d6eaf8", "#1a5276"),
          ("Hidden 2\n64  ReLU",  0.56, "#d5f5e3", "#1e8449"),
          ("Output\n5  Logits",   0.80, "#fde8d8", "#a04000")]

for (lbl, x, fc, tc) in layers:
    ax_arch.add_patch(FancyBboxPatch((x-0.1, 0.42), 0.18, 0.4,
                                     boxstyle="round,pad=0.02",
                                     facecolor=fc, edgecolor=tc, linewidth=2,
                                     transform=ax_arch.transAxes))
    ax_arch.text(x, 0.62, lbl, transform=ax_arch.transAxes,
                 ha="center", va="center", fontsize=11, fontweight="bold", color=tc)

# arrows
for x in [0.18, 0.42, 0.66]:
    ax_arch.annotate("", xy=(x+0.04, 0.62), xytext=(x, 0.62),
                     xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))

ax_arch.text(0.5, 0.33,
             "He Init:  W ~ N(0, √(2/n_in))     b = 0\n"
             "Loss:  L = −z_y + logsumexp(z)      (per sample)\n"
             "Update:  v = βv + (1−β)(g + λP)     P = P − αv",
             transform=ax_arch.transAxes, ha="center", va="top",
             fontsize=10.5, color="#2c3e50", linespacing=1.8,
             fontfamily="monospace",
             bbox=dict(facecolor="#f4f6f7", edgecolor="#aab7b8",
                       boxstyle="round,pad=0.4"))
ax_arch.text(0.5, 0.06,
             "β = 0.9   λ = 1e-4   B = 128   Epochs = 30",
             transform=ax_arch.transAxes, ha="center", va="bottom",
             fontsize=11, color=NAVY, fontweight="bold")

# [1,1] LR comparison
ax_lr = card_ax(fig, gs[1, 1], "Learning-Rate Comparison (Validation Loss)", NAVY)
val_losses = {
    "α = 0.01": [0.4621,0.3835,0.3427,0.3148,0.2945,0.2766,0.2636,0.2508,0.2412,0.2326,
                 0.2257,0.2188,0.2131,0.2076,0.2043,0.2019,0.1955,0.1918,0.1881,0.1872,
                 0.1839,0.1795,0.1768,0.1742,0.1717,0.1695,0.1674,0.1650,0.1643,0.1615],
    "α = 0.10": [0.2539,0.1982,0.1821,0.1693,0.1619,0.1548,0.1489,0.1437,0.1395,0.1340,
                 0.1289,0.1253,0.1210,0.1171,0.1140,0.1109,0.1082,0.1050,0.1026,0.1004,
                 0.0983,0.0964,0.0944,0.0924,0.0905,0.0888,0.0873,0.0885,0.0878,0.0876],
    "α = 0.50": [0.1705,0.1422,0.1208,0.1275,0.1146,0.1193,0.1092,0.0979,0.0916,0.0887,
                 0.0908,0.0835,0.0845,0.0861,0.0970,0.0816,0.0769,0.0784,0.0818,0.0819,
                 0.0747,0.0764,0.0772,0.0831,0.0787,0.0817,0.0803,0.0801,0.0756,0.0777],
}
ep = range(1, 31)
clrs = ["#3498db","#e67e22","#e74c3c"]
for (lbl, vals), c in zip(val_losses.items(), clrs):
    lw = 2.5 if "0.50" in lbl else 1.5
    ax_lr.plot(ep, vals, label=lbl, color=c, linewidth=lw,
               linestyle="-" if "0.50" in lbl else "--")
ax_lr.axvline(21, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.8)
ax_lr.text(21.5, 0.38, "Best\nepoch 21", fontsize=8.5, color="#e74c3c")
ax_lr.set_xlabel("Epoch", fontsize=11)
ax_lr.set_ylabel("Val Loss", fontsize=11)
ax_lr.legend(fontsize=10)
ax_lr.spines[["top","right"]].set_visible(False)
ax_lr.set_facecolor(WHITE)

# [2:,1] loss + accuracy curves side by side
ax_lc = card_ax(fig, gs[2, 1], "Training & Validation Loss  (α = 0.5)", NAVY)
train_loss = [0.1724,0.1393,0.1147,0.1179,0.1057,0.1084,0.0933,0.0852,0.0794,0.0753,
              0.0767,0.0693,0.0682,0.0720,0.0794,0.0649,0.0608,0.0604,0.0637,0.0619,
              0.0548,0.0566,0.0569,0.0628,0.0600,0.0568,0.0549,0.0591,0.0553,0.0531]
val_loss_b  = val_losses["α = 0.50"]
ax_lc.plot(ep, train_loss, label="Train", color="#3498db", linewidth=2)
ax_lc.plot(ep, val_loss_b, label="Val",   color="#e74c3c", linewidth=2)
ax_lc.axvline(21, color=ACCENT, linestyle=":", alpha=0.7)
ax_lc.set_xlabel("Epoch", fontsize=10); ax_lc.set_ylabel("Loss", fontsize=10)
ax_lc.legend(fontsize=9); ax_lc.spines[["top","right"]].set_visible(False)
ax_lc.set_facecolor(WHITE)

ax_ac = card_ax(fig, gs[3, 1], "Training & Validation Accuracy  (α = 0.5)", NAVY)
train_acc = [0.9515,0.9626,0.9666,0.9650,0.9695,0.9707,0.9731,0.9752,0.9772,0.9784,
             0.9778,0.9792,0.9793,0.9786,0.9766,0.9798,0.9815,0.9812,0.9803,0.9813,
             0.9834,0.9828,0.9822,0.9805,0.9813,0.9825,0.9825,0.9814,0.9838,0.9834]
val_acc =   [0.9510,0.9626,0.9658,0.9638,0.9688,0.9669,0.9689,0.9732,0.9742,0.9742,
             0.9750,0.9764,0.9761,0.9758,0.9714,0.9774,0.9793,0.9780,0.9762,0.9772,
             0.9794,0.9797,0.9780,0.9774,0.9767,0.9785,0.9785,0.9766,0.9792,0.9784]
ax_ac.plot(ep, [v*100 for v in train_acc], label="Train", color="#3498db", linewidth=2)
ax_ac.plot(ep, [v*100 for v in val_acc],   label="Val",   color="#e74c3c", linewidth=2)
ax_ac.axvline(21, color=ACCENT, linestyle=":", alpha=0.7)
ax_ac.set_xlabel("Epoch", fontsize=10); ax_ac.set_ylabel("Accuracy (%)", fontsize=10)
ax_ac.legend(fontsize=9); ax_ac.spines[["top","right"]].set_visible(False)
ax_ac.set_facecolor(WHITE)

# ══════════════════════════════════════════════════════════
# COLUMN 3: Results & Analysis
# ══════════════════════════════════════════════════════════

# [0,2] key metrics box
ax_met = fig.add_subplot(gs[0, 2])
ax_met.set_facecolor(WHITE)
ax_met.set_axis_off()
ax_met.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor="#dee2e6",
                                 linewidth=1.5, transform=ax_met.transAxes))
ax_met.text(0.5, 0.95, "Key Results", transform=ax_met.transAxes,
            ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)

metrics = [
    ("Test Accuracy",     "97.80%", "#2ecc71"),
    ("Balanced Accuracy", "85.34%", "#3498db"),
    ('Always-"N" Baseline', "82.76%", "#e74c3c"),
    ("Parameters",        "32,645", "#9b59b6"),
    ("Chosen α",          "0.5",    "#f39c12"),
    ("Best Epoch",        "21",     "#1abc9c"),
    ("Grad Check (W3)",   "1.49e-8 ✓", "#27ae60"),
    ("Grad Check (b3)",   "5.96e-8 ✓", "#27ae60"),
]
for i, (lbl, val, col) in enumerate(metrics):
    y = 0.83 - i * 0.105
    ax_met.add_patch(FancyBboxPatch((0.04, y-0.03), 0.92, 0.09,
                                    boxstyle="round,pad=0.01",
                                    facecolor=col+"18", edgecolor=col+"55",
                                    linewidth=1, transform=ax_met.transAxes))
    ax_met.text(0.08, y+0.015, lbl, transform=ax_met.transAxes,
                ha="left", va="center", fontsize=10, color="#2c3e50")
    ax_met.text(0.92, y+0.015, val, transform=ax_met.transAxes,
                ha="right", va="center", fontsize=11, fontweight="bold", color=col)

# [1,2] per-class accuracy
ax_pca = card_ax(fig, gs[1, 2], "Per-class Accuracy", NAVY)
pca = [99.54, 68.53, 91.37, 70.37, 96.89]
bars = ax_pca.bar(CLASS_NAMES, pca, color=CLASS_COLORS, width=0.55, edgecolor="white", linewidth=1.2)
ax_pca.axhline(85.34, color=NAVY,   linestyle="--", linewidth=1.5, label="Balanced acc 85.34%")
ax_pca.axhline(97.80, color=ACCENT, linestyle=":",  linewidth=1.5, label="Overall acc 97.80%")
ax_pca.set_ylim(0, 115)
ax_pca.set_ylabel("Accuracy (%)", fontsize=11)
for bar, v in zip(bars, pca):
    ax_pca.text(bar.get_x()+bar.get_width()/2, v+1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax_pca.legend(fontsize=9)
ax_pca.spines[["top","right"]].set_visible(False)
ax_pca.set_facecolor(WHITE)

# [2:, 2] confusion matrix
ax_cm = card_ax(fig, gs[2:, 2], "Test Confusion Matrix", NAVY)
ax_cm.set_axis_off()
img_path = os.path.join(RES, "confusion_matrix.png")
if os.path.exists(img_path):
    img = plt.imread(img_path)
    ax_cm.imshow(img, aspect="auto", extent=[0,1,0,1], transform=ax_cm.transAxes)
    ax_cm.set_facecolor(WHITE)

# ── FOOTER ──────────────────────────────────────────────────────────────────
foot = fig.add_axes([0.01, 0.005, 0.98, 0.022])
foot.set_axis_off()
foot.add_patch(FancyBboxPatch((0,0), 1, 1, boxstyle="round,pad=0.01",
                               facecolor=NAVY, edgecolor="none",
                               transform=foot.transAxes))
foot.text(0.02, 0.5,
          "Allowed: torch tensors, nn.Module, nn.Parameter, torch.logsumexp, numpy, pandas, matplotlib, Streamlit",
          transform=foot.transAxes, ha="left", va="center",
          fontsize=9.5, color="#aed6f1")
foot.text(0.98, 0.5,
          "NOT allowed: nn.Linear, nn.Sequential, nn.ReLU, torch.optim, nn.CrossEntropyLoss, scikit-learn",
          transform=foot.transAxes, ha="right", va="center",
          fontsize=9.5, color="#f1948a")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=NAVY)
print(f"Poster saved → {OUT}")
plt.close(fig)
