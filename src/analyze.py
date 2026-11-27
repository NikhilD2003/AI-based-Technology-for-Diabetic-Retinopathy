"""All tables, statistical tests and figures, computed only from results/preds.

    py -3.10 src/analyze.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, rankdata
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score, roc_curve

from data import ROOT

OUT = ROOT / "results"
FIG = OUT / "figures"
NAMES = ["full", "no_preproc", "no_aug", "no_class_weight", "size320", "resnet50", "scratch_cnn", "svm_rbf", "logreg"]
N_BOOT = 2000


def qwk(y, p, k=5):
    """Quadratic weighted kappa: 1 - sum(w*O) / sum(w*E), w_ij = (i-j)^2 / (k-1)^2."""
    O = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    i = np.arange(k)
    w = (i[:, None] - i[None, :]) ** 2 / (k - 1) ** 2
    return 1 - (w * O).sum() / (w * E).sum()


def metrics(y, probs):
    pred = probs.argmax(1)
    ref, score = y >= 2, probs[:, 2:].sum(1)  # referable DR = grade 2 or worse
    flag = score >= 0.5
    return dict(accuracy=accuracy_score(y, pred), qwk=qwk(y, pred),
                macro_f1=f1_score(y, pred, average="macro", labels=range(5), zero_division=0),
                macro_auc=roc_auc_score(y, probs / probs.sum(1, keepdims=True), multi_class="ovr"),
                referable_auc=roc_auc_score(ref, score),
                sensitivity=flag[ref].mean(), specificity=(~flag[~ref]).mean())


def delong(ref, a, b):
    """DeLong test for two correlated AUCs. Returns auc_a, auc_b, two-sided p."""
    scores = np.stack([a, b])
    pos, neg = scores[:, ref], scores[:, ~ref]
    m, n = pos.shape[1], neg.shape[1]
    ranks = rankdata(np.hstack([pos, neg]), axis=1)
    v10 = (ranks[:, :m] - rankdata(pos, axis=1)) / n          # structural components
    v01 = 1 - (ranks[:, m:] - rankdata(neg, axis=1)) / m
    auc = v10.mean(1)
    S = np.cov(v10) / m + np.cov(v01) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    p = 2 * norm.sf(abs(auc[0] - auc[1]) / np.sqrt(var)) if var > 0 else 1.0
    return auc[0], auc[1], p


def mcnemar(y, pa, pb):
    """Exact McNemar test on the discordant pairs."""
    b = int(((pa == y) & (pb != y)).sum())
    c = int(((pa != y) & (pb == y)).sum())
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def boot_ci(n, stat, seed=0):
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return np.percentile(vals, [2.5, 97.5])


def load(name):
    files = sorted((OUT / "preds").glob(f"{name}_s*.npz"))
    assert files, f"no predictions for {name}"
    return [np.load(f)["probs"] for f in files]


def md(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return "\n".join([head] + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(OUT / "images.csv")
    y = t.diagnosis.values
    mask = lambda ds, split=None: ((t.dataset == ds) & ((t.split == split) if split else True)).values
    evals = {"aptos_test": mask("aptos", "test"), "idrid_all": mask("idrid")}
    runs = {n: load(n) for n in NAMES}
    ens = {n: np.mean(r, axis=0) for n, r in runs.items()}  # seed-averaged probabilities, used for the tests

    # 1. per-seed metrics, then mean +/- SD
    rows = [dict(model=n, eval=e, seed=s, **metrics(y[m], p[m]))
            for n in NAMES for e, m in evals.items() for s, p in enumerate(runs[n])]
    rev = {"idrid_test": mask("idrid", "test"), "aptos_test": evals["aptos_test"]}
    rows += [dict(model="idrid_source", eval=e, seed=s, **metrics(y[m], p[m]))
             for e, m in rev.items() for s, p in enumerate(load("idrid_source"))]
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "metrics_per_seed.csv", index=False)
    g = per_seed.drop(columns="seed").groupby(["model", "eval"], sort=False)
    mean, sd, count = g.mean(), g.std(), g.size()
    summary = mean.apply(lambda col: [f"{v:.3f}" + (f" ± {sd.loc[i, col.name]:.3f}" if count[i] > 1 else "")
                                      for i, v in col.items()]).reset_index()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)

    # 2. bootstrap CIs, and every variant against the full model on the same images
    ci_rows, tests = [], []
    for e, m in evals.items():
        yy, ref = y[m], y[m] >= 2
        base = ens["full"][m]
        for n in NAMES:
            p = ens[n][m]
            pred = p.argmax(1)
            lo, hi = boot_ci(len(yy), lambda i: qwk(yy[i], pred[i]))
            ci_rows.append(dict(eval=e, model=n, qwk=round(qwk(yy, pred), 3), ci95=f"[{lo:.3f}, {hi:.3f}]"))
            if n == "full":
                continue
            bp = base.argmax(1)
            dlo, dhi = boot_ci(len(yy), lambda i: qwk(yy[i], bp[i]) - qwk(yy[i], pred[i]))
            auc_full, auc_other, p_auc = delong(ref, base[:, 2:].sum(1), p[:, 2:].sum(1))
            assert np.isclose(auc_full, roc_auc_score(ref, base[:, 2:].sum(1)))
            tests.append(dict(eval=e, comparison=f"full vs {n}", qwk_diff=qwk(yy, bp) - qwk(yy, pred),
                              qwk_diff_ci95=f"[{dlo:.3f}, {dhi:.3f}]",
                              accuracy_diff=(bp == yy).mean() - (pred == yy).mean(), mcnemar_p=mcnemar(yy, bp, pred),
                              auc_diff=auc_full - auc_other, delong_p=p_auc))
    ci, tests = pd.DataFrame(ci_rows), pd.DataFrame(tests)
    for col in ("mcnemar_p", "delong_p"):  # corrected within each evaluation set
        tests[col + "_holm"] = tests.groupby("eval")[col].transform(lambda s: holm(s.values))
    ci.to_csv(OUT / "bootstrap_ci.csv", index=False)
    tests.to_csv(OUT / "significance_tests.csv", index=False)

    # 3. cross-dataset drop, both directions (per seed, then averaged)
    cross = []
    for direction, name, inside, outside in [("APTOS -> IDRiD", "full", "aptos_test", "idrid_all"),
                                             ("IDRiD -> APTOS", "idrid_source", "idrid_test", "aptos_test")]:
        s = per_seed[per_seed.model == name].set_index(["eval", "seed"])
        for metric in ("qwk", "accuracy", "macro_f1", "referable_auc", "sensitivity", "specificity"):
            a, b = s.loc[inside, metric], s.loc[outside, metric]
            cross.append(dict(direction=direction, metric=metric, same_dataset=round(a.mean(), 3),
                              other_dataset=round(b.mean(), 3), drop=round((a - b).mean(), 3)))
    cross = pd.DataFrame(cross)
    cross.to_csv(OUT / "cross_dataset.csv", index=False)

    # 4. figures
    for e, m in evals.items():
        d = ConfusionMatrixDisplay.from_predictions(y[m], ens["full"][m].argmax(1), labels=range(5), cmap="Blues")
        d.ax_.set_title(f"EfficientNet-B0 (3-seed ensemble) on {e}")
        d.figure_.savefig(FIG / f"confusion_{e}.png", dpi=200, bbox_inches="tight")
        plt.close(d.figure_)

        plt.figure(figsize=(6, 5))
        for n in ("full", "resnet50", "scratch_cnn", "svm_rbf", "logreg"):
            score = ens[n][m, 2:].sum(1)
            fpr, tpr, _ = roc_curve(y[m] >= 2, score)
            plt.plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(y[m] >= 2, score):.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8)
        plt.xlabel("False positive rate"), plt.ylabel("True positive rate"), plt.title(f"Referable DR, {e}")
        plt.legend()
        plt.savefig(FIG / f"roc_{e}.png", dpi=200, bbox_inches="tight")
        plt.close()

    q = per_seed[per_seed["eval"].isin(list(evals)) & per_seed.model.isin(NAMES)].groupby(["model", "eval"]).qwk
    ax = q.mean().unstack().loc[NAMES].plot.bar(yerr=q.std().unstack().loc[NAMES].fillna(0), figsize=(10, 5), capsize=3)
    ax.set_ylabel("Quadratic weighted kappa"), ax.set_xlabel(""), ax.set_title("Ablation and model comparison (mean ± SD)")
    plt.xticks(rotation=30, ha="right")
    plt.savefig(FIG / "ablation_qwk.png", dpi=200, bbox_inches="tight")
    plt.close()

    h = pd.DataFrame(json.loads((OUT / "preds/full_s0.json").read_text())["history"]).set_index("epoch")
    ax = h.plot(subplots=True, figsize=(6, 5))
    ax[0].figure.savefig(FIG / "training_curve.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    # 5. one readable report
    tests_fmt = tests.copy()
    for c in tests_fmt.select_dtypes("float"):
        tests_fmt[c] = tests_fmt[c].map(lambda v: f"{v:.3g}")
    text = ["# Results (generated by `src/analyze.py`; do not edit)",
            "Seed-trained models show mean ± SD over 3 seeds. `logreg` and `svm_rbf` are single fits. "
            "`idrid_source` is the reverse direction (trained on IDRiD).",
            "## Metrics", md(summary),
            f"## QWK of seed-averaged predictions with bootstrap 95% CI ({N_BOOT} resamples)", md(ci),
            "## Full model vs each variant (paired tests, seed-averaged predictions; differences are full minus variant; "
            "columns ending in _holm are Holm-corrected within each evaluation set, the QWK intervals are not)", md(tests_fmt),
            "## Cross-dataset validation", md(cross)]
    (OUT / "SUMMARY.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print("wrote results/SUMMARY.md")


def self_check():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 300), rng.integers(0, 5, 300)
    assert np.isclose(qwk(a, b), cohen_kappa_score(a, b, weights="quadratic"))
    ref, s1, s2 = a >= 2, rng.random(300), rng.random(300)
    x1, x2, p = delong(ref, s1, s2)
    assert np.isclose(x1, roc_auc_score(ref, s1)) and np.isclose(x2, roc_auc_score(ref, s2)) and 0 <= p <= 1
    assert delong(ref, s1, s1)[2] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    self_check()
    main()
