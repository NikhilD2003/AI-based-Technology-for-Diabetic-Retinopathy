"""Grad-CAM for the seed-0 EfficientNet-B0: one correctly graded test image per grade and dataset.

    py -3.10 src/gradcam.py
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import CACHE, ROOT
from train import BASE, MODELS, PREDS, batch_of, build


def gradcam(net, x, cls):
    """cam = ReLU(sum_k alpha_k A_k), alpha_k = spatial mean of d(logit_cls)/dA_k, A = last conv maps."""
    maps = net.forward_features(x)
    grad = torch.autograd.grad(net.forward_head(maps)[0, cls], maps)[0]
    cam = F.relu((grad.mean((2, 3), keepdim=True) * maps).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[-2:], mode="bilinear")[0, 0]
    return (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()


if __name__ == "__main__":
    t = pd.read_csv(ROOT / "results/images.csv")
    images = torch.from_numpy(np.load(CACHE / "graham.npy")).cuda()
    net = build(BASE["model"]).cuda()
    net.load_state_dict(torch.load(MODELS / "full.pt", weights_only=True))
    net.eval()
    probs = np.load(PREDS / "full_s0.npz")["probs"]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for row, dataset in enumerate(["aptos", "idrid"]):
        for grade in range(5):
            m = (t.dataset == dataset) & (t.diagnosis == grade) & ((t.split == "test") | (dataset == "idrid"))
            ids = np.flatnonzero(m.values)
            correct = ids[probs[ids].argmax(1) == grade]
            # most confident correct example; if none was graded correctly, show a miss and say so
            i = int((correct if len(correct) else ids)[probs[correct if len(correct) else ids, grade].argmax()])
            pred = int(probs[i].argmax())
            x = batch_of(images, torch.tensor([i], device="cuda"), BASE["size"], False)
            cam = gradcam(net, x, pred)
            img = cv2.resize(images[i].cpu().numpy(), (BASE["size"],) * 2)
            heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            top, bottom = axes[2 * row, grade], axes[2 * row + 1, grade]
            top.imshow(img), bottom.imshow(cv2.addWeighted(img, 0.6, heat, 0.4, 0))
            top.set_title(f"{dataset}: true {grade}, predicted {pred}")
            top.axis("off"), bottom.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/gradcam.png", dpi=150)
    print("wrote results/figures/gradcam.png")
