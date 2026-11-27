"""Dataset tables and the preprocessed image cache.

    py -3.10 src/data.py      # builds data/cache/{plain,graham}.npy
"""
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SIZE = 320  # cached resolution; smaller model inputs are resized from it on the GPU
IDRID = ROOT / "data/IDRiD/B. Disease Grading"


def table():
    """One row per image: dataset, split, grade, path. Row order == cache order."""
    a = pd.read_csv(ROOT / "results/splits.csv")
    a["path"] = [f"data/APTOS/train_images/{i}.png" for i in a.id_code]
    a["dataset"] = "aptos"
    rows = [a]
    for split, folder, label in [("train", "a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
                                 ("test", "b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv")]:
        d = pd.read_csv(IDRID / "2. Groundtruths" / label)
        d = pd.DataFrame({"id_code": d["Image name"], "diagnosis": d["Retinopathy grade"]})
        d["path"] = [str((IDRID / "1. Original Images" / folder / f"{i}.jpg").relative_to(ROOT)) for i in d.id_code]
        d["split"], d["dataset"] = split, "idrid"
        rows.append(d)
    t = pd.concat(rows, ignore_index=True)
    # IDRiD ships no validation set: hold out a stratified 15% of its training set (for the reverse direction)
    from sklearn.model_selection import train_test_split
    pool = t.index[(t.dataset == "idrid") & (t.split == "train")]
    _, val = train_test_split(pool, test_size=0.15, stratify=t.diagnosis[pool], random_state=42)
    t.loc[val, "split"] = "val"
    return t


def crop_retina(img, tol=7):
    """Cut the black border around the fundus."""
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def square(img, size=SIZE):
    """Pad to square (keeps the retina round), then resize."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    img = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ben_graham(img):
    """I' = 4*I - 4*(G_sigma * I) + 128, then a circular mask to drop the rim artefact."""
    sigma = img.shape[0] / 30
    out = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    mask = np.zeros(img.shape[:2], np.uint8)
    r = img.shape[0] // 2
    cv2.circle(mask, (r, r), int(r * 0.95), 1, -1)
    return out * mask[..., None]


def process(path):
    img = cv2.imread(str(ROOT / path))
    assert img is not None, path
    plain = square(crop_retina(img))
    return cv2.cvtColor(plain, cv2.COLOR_BGR2RGB), cv2.cvtColor(ben_graham(plain), cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    t = table()
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        out = pool.map(process, t.path, chunksize=8)
    for k, name in enumerate(["plain", "graham"]):
        np.save(CACHE / f"{name}.npy", np.stack([o[k] for o in out]))
    t.to_csv(ROOT / "results/images.csv", index=False)
    print(t.groupby(["dataset", "split"]).size())
