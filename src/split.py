"""Deduplicated, stratified 70/15/15 split of the labelled APTOS images."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

df = pd.read_csv(ROOT / "data/APTOS/train.csv")
df["md5"] = [hashlib.md5((ROOT / f"data/APTOS/train_images/{i}.png").read_bytes()).hexdigest()
             for i in df.id_code]
n_raw = len(df)
# identical files with different grades: the true label is unknown, so drop them all
# ponytail: exact copies only; near-duplicates (re-saved/cropped) need perceptual hashing
df = df[df.groupby("md5").diagnosis.transform("nunique") == 1]
df = df.drop_duplicates("md5")

train, rest = train_test_split(df, test_size=0.30, stratify=df.diagnosis, random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest.diagnosis, random_state=SEED)
out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])

assert out.md5.is_unique and len(out) == len(df)
out[["id_code", "diagnosis", "split"]].to_csv(ROOT / "results/splits.csv", index=False)
print(f"{n_raw} -> {len(out)} images")
print(pd.crosstab(out.split, out.diagnosis, margins=True))
