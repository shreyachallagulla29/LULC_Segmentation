"""
train_lib_r8.py  -  round 8: ignore label BORDERS during training (preprocessing step 1).

Pixels on the boundary between two classes are often wrong in WorldCover: they are mixed pixels
(half field, half road) or the label is shifted by a pixel. Here every training pixel whose
neighbourhood (a (2*BORDER_PX+1) x (2*BORDER_PX+1) window) contains another class is set to IGNORE,
so the loss never learns from these uncertain pixels.

* Only TRAINING labels are changed; validation and test are scored on the full, unchanged labels,
  so the scores stay directly comparable with v5 / v6 / v7.
* Everything else (model, loss, training loop, evaluation, predict_region) is taken unchanged
  from train_lib_r6.py, so the only difference to v6 is this step.
"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from train_lib_r6 import *                     # model, train, evaluate, report, predict_region, ...
from train_lib_r6 import TrainSeries, EvalSeries, DEV, IGNORE


def border_mask(y, r=1):
    """True where a pixel has a different label (or ignore) within r pixels: the class boundaries."""
    H, W = y.shape
    p = np.pad(y, r, mode="edge")
    m = np.zeros((H, W), bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                m |= p[r + dy:r + dy + H, r + dx:r + dx + W] != y
    return m


def ignore_borders(y, r=1):
    """Label copy with the class boundaries set to IGNORE (r = 0 -> unchanged)."""
    if not r:
        return y
    return np.where(border_mask(y, r), IGNORE, y).astype(y.dtype)


class TrainSeriesBorder(TrainSeries):
    """Same as TrainSeries (round 6) + label borders set to IGNORE before augmentation."""
    def __init__(self, files, data, noise_db=2.0, roll=True, month_drop=0.1, border_px=1):
        super().__init__(files, data, noise_db, roll, month_drop)
        self.border_px = border_px
    def __getitem__(self, k):
        d = np.load(self.data / self.files[k])
        x, y = d["x"].astype(np.float32), d["y"].astype(np.int64)
        y = ignore_borders(y, self.border_px)                               # <- the only new step
        x = x + np.random.uniform(-self.noise_db, self.noise_db, (1, 3, 1, 1)).astype(np.float32)
        if self.roll:
            x = np.roll(x, random.randrange(12), axis=0)
        m = random.randrange(12)
        if self.month_drop:
            drop = np.random.rand(12) < self.month_drop; drop[m] = False
            x[drop] = np.nan
        y = np.where(np.isnan(x[m]).any(0), IGNORE, y)
        if random.random() < 0.5: x, y = x[..., ::-1], y[:, ::-1]
        if random.random() < 0.5: x, y = x[..., ::-1, :], y[::-1]
        r = random.randrange(4); x, y = np.rot90(x, r, (2, 3)), np.rot90(y, r)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(m),
                torch.from_numpy(np.ascontiguousarray(y)))


def make_loaders_border(cfg, data, index, val_ds=None):
    train_ds = TrainSeriesBorder(index[index.split == "train"].file, data, cfg.get("NOISE_DB", 2.0),
                                 cfg.get("ROLL", True), cfg.get("MONTH_DROP", 0.1), cfg.get("BORDER_PX", 1))
    val_ds = val_ds or EvalSeries(index[index.split == "val"].file, data, "loading val")
    train_dl = DataLoader(train_ds, cfg["BATCH_SIZE"], shuffle=True, drop_last=True,
                          num_workers=cfg["NUM_WORKERS"], pin_memory=(DEV == "cuda"))
    val_dl = DataLoader(val_ds, cfg.get("EVAL_BATCH", 2), shuffle=False)
    print(f"train patches {len(train_ds)} (label borders of {cfg.get('BORDER_PX', 1)} px ignored) | "
          f"val patches {len(val_ds)} (full labels, x 12 months each)")
    return train_dl, val_dl


# ------------------------------------------------------------------ other segmentation networks (U-Net, DeepLabV3+)
# The temporal encoder, year statistics and fusion stay the same; only the 2-D network after the
# fusion (cfg["SEG_ARCH"]) is swapped. Without SEG_ARCH -> SegFormer, i.e. exactly v6 / v7 / v8.
SEG_ARCHS = {"segformer_mitb2": ("Segformer", "mit_b2"),
             "unet_resnet34": ("Unet", "resnet34"),
             "deeplabv3plus_resnet50": ("DeepLabV3Plus", "resnet50")}


def build_model(cfg, mstats, ystats, weights="imagenet"):
    import segmentation_models_pytorch as smp
    arch = cfg.get("SEG_ARCH", "segformer_mitb2")
    fuse = cfg.get("FUSE_DIM", 32)
    model = TemporalSegFormer(mstats, ystats, cfg.get("TEMP_DIM", 24), fuse,
                              weights if arch == "segformer_mitb2" else None)
    if arch != "segformer_mitb2":
        cls, enc = SEG_ARCHS[arch]
        model.seg = getattr(smp, cls)(encoder_name=enc, encoder_weights=weights, in_channels=fuse, classes=NC)
    return model
