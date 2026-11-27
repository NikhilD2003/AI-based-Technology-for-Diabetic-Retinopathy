"""Descriptive figures for the report: class distribution, preprocessing and cross-dataset.

    py -3.10 src/figures.py     # after analyze.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CACHE, ROOT

OUT = ROOT / "results"
FIG = OUT / "figures"


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


t = pd.read_csv(OUT / "images.csv")

counts = pd.crosstab(t.diagnosis, t.dataset.str.upper())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, name in zip(axes, counts):
    ax.bar_label(ax.bar(counts.index, counts[name]))
    ax.set_title(f"{name} ({counts[name].sum()} images)"), ax.set_xlabel("DR grade"), ax.set_ylabel("Images")
    ax.margins(y=0.12)
save(fig, "class_distribution.png")

plain, graham = np.load(CACHE / "plain.npy", mmap_mode="r"), np.load(CACHE / "graham.npy", mmap_mode="r")
picks = [int(t.index[(t.dataset == d) & (t.diagnosis == g)][0]) for d, g in
         [("aptos", 0), ("aptos", 2), ("aptos", 4), ("idrid", 0), ("idrid", 2), ("idrid", 4)]]
fig, axes = plt.subplots(2, len(picks), figsize=(15, 5.4))
for col, i in enumerate(picks):
    for row, (label, cache) in enumerate([("crop + resize", plain), ("+ Ben Graham", graham)]):
        axes[row, col].imshow(cache[i])
        axes[row, col].set_title(f"{t.dataset[i].upper()} grade {t.diagnosis[i]}, {label}", fontsize=9)
        axes[row, col].axis("off")
fig.tight_layout()
save(fig, "preprocessing.png")

m = pd.read_csv(OUT / "metrics_per_seed.csv")
panels = [("APTOS -> IDRiD", "full", ["aptos_test", "idrid_all"], ["APTOS test", "IDRiD (all)"]),
          ("IDRiD -> APTOS", "idrid_source", ["idrid_test", "aptos_test"], ["IDRiD test", "APTOS test"])]
metrics = ["qwk", "accuracy", "macro_f1", "referable_auc"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (title, model, evals, labels) in zip(axes, panels):
    for k, (e, label) in enumerate(zip(evals, labels)):
        d = m[(m.model == model) & (m["eval"] == e)][metrics]
        ax.bar(np.arange(len(metrics)) + (k - 0.5) * 0.38, d.mean(), 0.38, yerr=d.std(), capsize=3,
               label=f"{'same' if k == 0 else 'other'} dataset: {label}")
    ax.set_xticks(range(len(metrics)), ["QWK", "Accuracy", "Macro F1", "Referable AUC"])
    ax.set_title(f"Train {title.replace('->', '→ test')}"), ax.set_ylim(0, 1.15), ax.legend(fontsize=8, loc="upper left")
save(fig, "cross_dataset.png")

print("figures written")
