"""Classical baselines on hand-crafted features: logistic regression and RBF SVM.

    py -3.10 src/baseline.py
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import CACHE, ROOT


def lbp_hist(g):
    """8-neighbour local binary pattern histogram (texture)."""
    c = g[1:-1, 1:-1]
    shifts = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    code = sum((g[y:y + c.shape[0], x:x + c.shape[1]] >= c).astype(np.int32) << b for b, (y, x) in enumerate(shifts))
    return np.bincount(code[c > 0].ravel(), minlength=256) / max((c > 0).sum(), 1)


def features(img):
    """Colour mean/std/histograms, edge-orientation histogram and LBP texture inside the retina."""
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 0
    px = img[mask].astype(np.float32)
    colour = [px.mean(0), px.std(0)] + [np.histogram(px[:, c], 32, (0, 256))[0] / len(px) for c in range(3)]
    green = img[..., 1]  # lesions have the most contrast in the green channel
    gx, gy = cv2.Sobel(green, cv2.CV_32F, 1, 0), cv2.Sobel(green, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    edges = np.histogram(ang[mask] % np.pi, 16, (0, np.pi), weights=mag[mask])[0] / len(px)
    return np.concatenate(colour + [edges, lbp_hist(green)])


if __name__ == "__main__":
    t = pd.read_csv(ROOT / "results/images.csv")
    X = np.stack([features(i) for i in np.load(CACHE / "graham.npy")])
    train = ((t.dataset == "aptos") & (t.split == "train")).values
    val = ((t.dataset == "aptos") & (t.split == "val")).values
    y = t.diagnosis.values
    models = {
        "logreg": [LogisticRegression(C=c, max_iter=5000, class_weight="balanced") for c in (0.01, 0.1, 1, 10)],
        "svm_rbf": [SVC(C=c, class_weight="balanced", probability=True, random_state=0) for c in (0.5, 2, 8)],
    }
    (ROOT / "results/preds").mkdir(exist_ok=True)
    for name, candidates in models.items():
        fitted = [make_pipeline(StandardScaler(), m).fit(X[train], y[train]) for m in candidates]
        # C is chosen on the validation split; test and IDRiD are never used for selection
        score = [cohen_kappa_score(y[val], m.predict_proba(X[val]).argmax(1), weights="quadratic") for m in fitted]
        best = fitted[int(np.argmax(score))]
        np.savez_compressed(ROOT / f"results/preds/{name}_s0.npz", probs=best.predict_proba(X))
        print(f"{name}: val QWK {max(score):.4f} with C={best[-1].C}")
