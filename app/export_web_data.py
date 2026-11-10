"""
Export model predictions + ECG signals + result images → web_data.json
Used by the standalone HTML website.
"""
import torch, json, base64, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ECGNet
from prepare import prepare

RES  = os.path.join(os.path.dirname(__file__), "..", "results")
OUT  = os.path.join(os.path.dirname(__file__), "web_data.js")

def img_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = prepare()

print("Loading model...")
torch.manual_seed(0)
model = ECGNet()
model.load_state_dict(torch.load(os.path.join(RES,"best_model.pth"),
                                  weights_only=True, map_location="cpu"))
model.eval()

# --- predictions for ALL test rows ---
print("Running predictions on test set...")
with torch.no_grad():
    logits = model(X_test)
    probs_all = torch.softmax(logits, dim=1).numpy()
    preds_all = logits.argmax(dim=1).numpy()

# --- sample 60 signals per class (300 total) for the predictor panel ---
print("Sampling signals...")
CLASS_SIGNALS = {}
for c in range(5):
    idx = (y_test == c).nonzero(as_tuple=True)[0].numpy()
    chosen = idx[:60]           # first 60 of each class
    CLASS_SIGNALS[c] = {
        "signals": X_test[chosen, :].numpy().tolist(),
        "row_ids": chosen.tolist(),
    }

# --- build compact prediction lookup ---
pred_lookup = []
for i in range(len(y_test)):
    pred_lookup.append({
        "t": int(y_test[i].item()),
        "p": int(preds_all[i]),
        "pr": [round(float(v), 4) for v in probs_all[i]],
    })

# --- encode result images ---
print("Encoding images...")
images = {
    "example_beats":    img_b64(os.path.join(RES, "example_beats.png")),
    "lr_comparison":    img_b64(os.path.join(RES, "lr_comparison.png")),
    "loss_curves":      img_b64(os.path.join(RES, "loss_curves.png")),
    "accuracy_curves":  img_b64(os.path.join(RES, "accuracy_curves.png")),
    "confusion_matrix": img_b64(os.path.join(RES, "confusion_matrix.png")),
    "poster":           img_b64(os.path.join(RES, "poster.png")),
}

data = {
    "predictions": pred_lookup,
    "class_signals": CLASS_SIGNALS,
    "images": images,
}

print(f"Writing {OUT} ...")
with open(OUT, "w") as f:
    f.write("const WEB_DATA = ")
    json.dump(data, f, separators=(',', ':'))
    f.write(";")

kb = os.path.getsize(OUT) / 1024
print(f"Done — {kb:.0f} KB")
