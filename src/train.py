"""Train every experiment and save softmax probabilities for all images.

    py -3.10 src/train.py                      # all runs; finished runs are skipped
    py -3.10 src/train.py --only full --seeds 0
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from data import CACHE, ROOT

PREDS = ROOT / "results/preds"
MODELS = ROOT / "results/models"
SEEDS = [0, 1, 2]

BASE = dict(model="efficientnet_b0", prep="graham", aug=True, weighted=True, size=224,
            source="aptos", epochs=12, lr=5e-4, batch=32)
# ablation = BASE with exactly one thing changed
EXPERIMENTS = {
    "full": {},
    "no_preproc": dict(prep="plain"),
    "no_aug": dict(aug=False),
    "no_class_weight": dict(weighted=False),
    "size320": dict(size=320),
    "resnet50": dict(model="resnet50", lr=2e-4),
    "scratch_cnn": dict(model="scratch", lr=2e-3, epochs=25),   # no pretrained weights, so more epochs
    "idrid_source": dict(source="idrid", epochs=30),            # reverse cross-dataset direction
}


class ScratchCNN(nn.Module):
    """5 x (conv 3x3 -> batch norm -> ReLU -> max pool), global average pool, linear."""
    def __init__(self, classes=5):
        super().__init__()
        layers, c = [], 3
        for out in (32, 64, 128, 256, 256):
            layers += [nn.Conv2d(c, out, 3, padding=1, bias=False), nn.BatchNorm2d(out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(name):
    return ScratchCNN() if name == "scratch" else timm.create_model(name, pretrained=True, num_classes=5)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def batch_of(images, idx, size, aug):
    """images: uint8 N,H,W,3 on the GPU. Resize and augmentation also run on the GPU."""
    x = images[idx].permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != size:
        x = F.interpolate(x, size=size, mode="bilinear", antialias=True)
    if aug:
        n, d = len(x), x.device
        angle = torch.rand(n, device=d) * 2 * torch.pi          # a fundus has no preferred orientation
        scale = 1 + (torch.rand(n, device=d) - 0.5) * 0.2
        flip = torch.randint(0, 2, (n,), device=d) * 2 - 1
        theta = torch.zeros(n, 2, 3, device=d)
        theta[:, 0, 0], theta[:, 0, 1] = angle.cos() * scale * flip, -angle.sin() * scale
        theta[:, 1, 0], theta[:, 1, 1] = angle.sin() * scale * flip, angle.cos() * scale
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
        bright = 1 + (torch.rand(n, 1, 1, 1, device=d) - 0.5) * 0.3
        contrast = 1 + (torch.rand(n, 1, 1, 1, device=d) - 0.5) * 0.3
        m = x.mean((2, 3), keepdim=True)
        x = (((x - m) * contrast + m) * bright).clamp(0, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(net, images, idx, size):
    net.eval()
    out = []
    for i in range(0, len(idx), 64):
        with torch.autocast("cuda"):
            logits = net(batch_of(images, idx[i:i + 64], size, False))
        out.append(logits.float().softmax(1).cpu())
    return torch.cat(out).numpy()


def run(name, seed, t, caches):
    cfg = {**BASE, **EXPERIMENTS[name]}
    out = PREDS / f"{name}_s{seed}.npz"
    if out.exists():
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = time.time()
    images = caches[cfg["prep"]]
    y = torch.tensor(t.diagnosis.values, device="cuda")
    ids = lambda split: torch.tensor(np.flatnonzero((t.dataset == cfg["source"]) & (t.split == split)), device="cuda")
    train, val = ids("train"), ids("val")

    net = build(cfg["model"]).cuda()
    counts = torch.bincount(y[train], minlength=5).float()
    weight = len(train) / (5 * counts) if cfg["weighted"] else None   # w_c = N / (K * n_c)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    steps = len(train) // cfg["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps * cfg["epochs"], pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda")

    best, history = (-2.0, None), []
    for epoch in range(cfg["epochs"]):
        net.train()
        perm = train[torch.randperm(len(train), device="cuda")]
        total = 0.0
        for s in range(steps):
            idx = perm[s * cfg["batch"]:(s + 1) * cfg["batch"]]
            x = batch_of(images, idx, cfg["size"], cfg["aug"])
            with torch.autocast("cuda"):
                loss = loss_fn(net(x), y[idx])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        p = predict(net, images, val, cfg["size"])
        kappa = cohen_kappa_score(y[val].cpu(), p.argmax(1), weights="quadratic", labels=range(5))
        history.append(dict(epoch=epoch + 1, train_loss=total / steps, val_qwk=kappa))
        if kappa > best[0]:  # model selection uses the source validation split only
            best = (kappa, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    net.load_state_dict(best[1])
    probs = predict(net, images, torch.arange(len(t), device="cuda"), cfg["size"])
    PREDS.mkdir(parents=True, exist_ok=True)
    if seed == 0:
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(best[1], MODELS / f"{name}.pt")
    minutes = (time.time() - start) / 60
    (PREDS / f"{name}_s{seed}.json").write_text(json.dumps(dict(config=cfg, minutes=minutes, history=history), indent=1))
    np.savez_compressed(out, probs=probs)  # written last: its presence marks the run as done
    print(f"{name} seed {seed}: val QWK {best[0]:.4f} in {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    t = pd.read_csv(ROOT / "results/images.csv")
    caches = {k: torch.from_numpy(np.load(CACHE / f"{k}.npy")).cuda() for k in ("plain", "graham")}
    for seed in args.seeds:       # all experiments at seed 0 first, so a full table exists early
        for name in args.only:
            run(name, seed, t, caches)
    print("ALL DONE", flush=True)
