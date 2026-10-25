"""
train_lib_r5.py  -  rounds 4b and 5: context channels from the whole year OR a short window.

FEATURES = "monthly"          -> 3 channels: VV, VH, VV-VH of the month
FEATURES = "monthly+year"     -> 7 channels: + mean / std of VV and VH over all 12 months (as in r4)
FEATURES = "monthly+window3"  -> 7 channels: + mean / std of VV and VH over a 3-MONTH WINDOW
                                 (previous, current, next month; 2 months at Jan / Dec).
                                 Keeps seasonal context but still follows real change in time.
All context channels are computed from the 12 months stored in every patch.

Evaluation (EvalPatches) stores the clipped dB values and normalises on the fly, so ONE loaded
val/test set can serve models trained on different data folders (each with its own
normalisation statistics) and with different input features - needed for a fair comparison.

Based on train_lib_r4.py.

train_lib.py  -  shared training / evaluation code for the model comparison.

All three model notebooks (04a, 04b, 04c) import this file, so data loading,
augmentation, loss, metrics and the training loop are IDENTICAL for every
model. The only thing that differs between notebooks is the network.
This is what makes the comparison fair.
"""
import json, random, time, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from preprocess import CLASSES, IGNORE, load_stats

NC = len(CLASSES)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ setup
def setup(cfg):
    """Seeds, shared-memory fix, run folder, data index and norm stats."""
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")          # little /dev/shm on the server
    random.seed(cfg["SEED"]); np.random.seed(cfg["SEED"])
    torch.manual_seed(cfg["SEED"]); torch.cuda.manual_seed_all(cfg["SEED"])

    run = Path("runs") / cfg["RUN_NAME"]; run.mkdir(parents=True, exist_ok=True)
    data = Path(cfg["DATA"])
    stats = load_stats(data / "norm_stats.json")
    index = pd.read_csv(data / "patch_index.csv")
    index["file"] = index["file"].str.replace("\\", "/", regex=False)
    json.dump(cfg, open(run / "config.json", "w"), indent=2)

    print("device:", DEV, end="")
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f" | GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: no GPU - training will be very slow (start a GPU server).")
    return run, data, stats, index


# ------------------------------------------------------------------ data
def _prep(x, y, mean, std):
    y = np.where(np.isnan(x).any(0), IGNORE, y)          # no radar this month -> ignore
    x = np.nan_to_num((x - mean) / std, nan=0.0)
    return x.astype(np.float32), y.astype(np.int64)


# ------------------------------------------------------------------ year-context features
YEAR_NAMES = ["VV_mean", "VH_mean", "VV_std", "VH_std"]


def year_features(x12):
    """x12: (12, 3, H, W) clipped dB with NaN -> (4, H, W): mean and std over the
    12 months of VV and VH (NaN months ignored; all-NaN pixels stay NaN)."""
    import warnings
    v = x12[:, :2].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.concatenate([np.nanmean(v, 0), np.nanstd(v, 0)]).astype(np.float32)


def year_stats(index, data, path=None):
    """Mean / std of the 4 year channels over the TRAINING patches (cached as JSON)."""
    path = Path(path or Path(data) / "norm_stats_year.json")
    if path.exists():
        return json.load(open(path))
    n = np.zeros(4); s = np.zeros(4); ss = np.zeros(4)
    for f in tqdm(index[index.split == "train"].file, desc="year-channel stats"):
        yf = year_features(np.load(Path(data) / f)["x"].astype(np.float32)).astype(np.float64)
        for b in range(4):
            v = yf[b][~np.isnan(yf[b])]
            n[b] += v.size; s[b] += v.sum(); ss[b] += (v ** 2).sum()
    mean = s / n; std = np.sqrt(ss / n - mean ** 2)
    out = {"names": YEAR_NAMES, "mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
    json.dump(out, open(path, "w"), indent=2)
    return out


def _norm_year(yf, ystats):
    m = np.array(ystats["mean"], np.float32)[:, None, None]
    sd = np.array(ystats["std"], np.float32)[:, None, None]
    return np.nan_to_num((yf - m) / sd, nan=0.0).astype(np.float32)


def window_features(x12, m, half=1):
    """Mean / std of VV and VH over months m-half .. m+half (clipped to Jan..Dec) -> (4, H, W)."""
    return year_features(x12[max(0, m - half): min(12, m + half + 1)])


def feature_kind(cfg):
    f = cfg.get("FEATURES", "monthly")
    return {"monthly": None, "monthly+year": "year", "monthly+window3": "window3"}[f]


def feature_stats(index, data, kind):
    """Normalisation stats of the 4 context channels over the TRAINING patches (cached JSON)."""
    if kind is None:
        return None
    if kind == "year":
        return year_stats(index, data)
    path = Path(data) / f"norm_stats_{kind}.json"
    if path.exists():
        return json.load(open(path))
    n = np.zeros(4); s = np.zeros(4); ss = np.zeros(4)
    for f in tqdm(index[index.split == "train"].file, desc=f"{kind}-channel stats"):
        x12 = np.load(Path(data) / f)["x"].astype(np.float32)
        for m in range(12):
            wf = window_features(x12, m).astype(np.float64)
            for b in range(4):
                v = wf[b][~np.isnan(wf[b])]
                n[b] += v.size; s[b] += v.sum(); ss[b] += (v ** 2).sum()
    mean = s / n; std = np.sqrt(ss / n - mean ** 2)
    out = {"names": YEAR_NAMES, "kind": kind, "mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
    json.dump(out, open(path, "w"), indent=2)
    return out


def in_channels_of(cfg):
    return 3 if feature_kind(cfg) is None else 7


class TrainPatches(Dataset):
    """One random month per patch per epoch + flips / 90° rotations / ±NOISE_DB dB offset.
    month_mix: probability of averaging the chosen month with a second random month
    (a new, realistic-looking radar image; the label is the same for every month)."""
    def __init__(self, files, data, stats, noise_db=1.0, month_mix=0.0, fstats=None, kind=None):
        self.files, self.data = list(files), data
        self.noise_db, self.month_mix = noise_db, month_mix
        self.fstats, self.kind = fstats, kind     # kind None -> 3 channels; "year"/"window3" -> 7
        self.mean = np.array(stats["mean"], np.float32)[:, None, None]
        self.std = np.array(stats["std"], np.float32)[:, None, None]
    def __len__(self): return len(self.files)
    def __getitem__(self, k):
        d = np.load(self.data / self.files[k])
        m = random.randrange(12)
        x, y = d["x"][m].astype(np.float32), d["y"]
        if self.month_mix and random.random() < self.month_mix:
            x = 0.5 * (x + d["x"][random.randrange(12)].astype(np.float32))
        off = np.random.uniform(-self.noise_db, self.noise_db, (3, 1, 1)).astype(np.float32)
        x = x + off
        x, y = _prep(np.ascontiguousarray(x), np.ascontiguousarray(y), self.mean, self.std)
        if self.kind is not None:
            x12 = d["x"].astype(np.float32)
            yf = year_features(x12) if self.kind == "year" else window_features(x12, m)
            yf[:2] += off[:2]                      # same calibration offset on the context means
            x = np.concatenate([x, _norm_year(yf, self.fstats)])
        if random.random() < 0.5: x, y = x[:, :, ::-1], y[:, ::-1]
        if random.random() < 0.5: x, y = x[:, ::-1], y[::-1]
        r = random.randrange(4); x, y = np.rot90(x, r, (1, 2)), np.rot90(y, r)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class EvalPatches(Dataset):
    """All patches x all 12 months, no augmentation. Stores the CLIPPED dB values (half precision,
    NaN = no data) and normalises in __getitem__, so the same loaded set can be used for models
    with different normalisation statistics and input features.
    Call .configure(stats, kind, fstats) before use (make_loaders does this)."""
    def __init__(self, files, data, stats=None, desc="loading", kind=None, fstats=None):
        self.xs, self.ys, self.month = [], [], []
        for f in tqdm(list(files), desc=desc):
            d = np.load(data / f)
            for m in range(12):
                self.xs.append(torch.from_numpy(d["x"][m].astype(np.float16)))
                self.ys.append(torch.from_numpy(d["y"].astype(np.uint8)))
                self.month.append(m)
        self.configure(stats, kind, fstats)

    def configure(self, stats, kind=None, fstats=None):
        self.kind = kind
        if stats is not None:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32)[:, None, None]
            self.std = torch.tensor(stats["std"], dtype=torch.float32)[:, None, None]
        if kind is not None:
            self.fmean = torch.tensor(fstats["mean"], dtype=torch.float32)[:, None, None]
            self.fstd = torch.tensor(fstats["std"], dtype=torch.float32)[:, None, None]
        return self

    @staticmethod
    def _nanstats(v):                                   # v: (T, 2, H, W) -> (4, H, W)
        mean = torch.nanmean(v, 0)
        var = (torch.nanmean(v * v, 0) - mean * mean).clamp(min=0)
        return torch.cat([mean, var.sqrt()])

    def __len__(self): return len(self.xs)

    def __getitem__(self, k):
        raw = self.xs[k].float()
        y = self.ys[k].long()
        y = torch.where(torch.isnan(raw).any(0), torch.tensor(IGNORE), y)
        x = torch.nan_to_num((raw - self.mean) / self.std, nan=0.0)
        if self.kind is not None:
            p, m = divmod(k, 12)
            lo, hi = (0, 12) if self.kind == "year" else (max(0, m - 1), min(12, m + 2))
            v = torch.stack([self.xs[p * 12 + t][:2].float() for t in range(lo, hi)])
            f = torch.nan_to_num((self._nanstats(v) - self.fmean) / self.fstd, nan=0.0)
            x = torch.cat([x, f])
        return x, y


def make_loaders(cfg, data, stats, index, val_ds=None):
    """val_ds can be passed in to reuse an already loaded validation set; it is
    (re)configured here for this run's normalisation and input features."""
    kind = feature_kind(cfg)
    fstats = feature_stats(index, data, kind)
    train_ds = TrainPatches(index[index.split == "train"].file, data, stats,
                            noise_db=cfg.get("NOISE_DB", 1.0), month_mix=cfg.get("MONTH_MIX", 0.0),
                            fstats=fstats, kind=kind)
    if val_ds is None:
        val_ds = EvalPatches(index[index.split == "val"].file, data, desc="loading val")
    val_ds.configure(stats, kind, fstats)
    train_dl = DataLoader(train_ds, cfg["BATCH_SIZE"], shuffle=True, drop_last=True,
                          num_workers=cfg["NUM_WORKERS"], pin_memory=(DEV == "cuda"),
                          persistent_workers=cfg["NUM_WORKERS"] > 0)
    val_dl = DataLoader(val_ds, 32, shuffle=False)
    print(f"train patches {len(train_ds)} | val samples {len(val_ds)} "
          f"({len(val_ds) // 12} patches x 12 months)")
    return train_dl, val_dl


# ------------------------------------------------------------------ loss + metrics
def make_loss(cfg, stats):
    """Weighted cross-entropy (+ optional label smoothing) + a second term:
    LOSS = "dice" (default, as in round 1-2), "lovasz" (optimises IoU directly),
    "focal" (focuses on hard pixels) or "none"."""
    import segmentation_models_pytorch as smp
    cw = (torch.tensor(stats["class_weights"], dtype=torch.float32, device=DEV)
          if cfg.get("USE_CLASS_WEIGHTS", True) else None)
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=IGNORE,
                             label_smoothing=cfg.get("LABEL_SMOOTHING", 0.0))
    kind = cfg.get("LOSS", "dice" if cfg.get("USE_DICE", True) else "none")
    extra = {"dice": smp.losses.DiceLoss("multiclass", ignore_index=IGNORE),
             "lovasz": smp.losses.LovaszLoss("multiclass", ignore_index=IGNORE),
             "focal": smp.losses.FocalLoss("multiclass", ignore_index=IGNORE),
             "none": None}[kind]
    def loss_fn(out, y):
        out = out.float()
        return ce(out, y) + (extra(out, y) if extra is not None else 0.0)
    return loss_fn


def confusion(pred, y):
    k = y != IGNORE
    return torch.bincount(y[k] * NC + pred[k], minlength=NC * NC).reshape(NC, NC)


def scores(cm):
    cm = cm.double(); tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp(min=1)
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp(min=1)
    return (tp.sum() / cm.sum().clamp(min=1)).item(), iou.cpu().numpy(), f1.cpu().numpy()


@torch.no_grad()
def evaluate(model, dl, loss_fn=None):
    model.eval(); total, n = 0.0, 0
    cm = torch.zeros(NC, NC, dtype=torch.long, device=DEV)
    for x, y in dl:
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            out = model(x)
        if loss_fn is not None:
            total += loss_fn(out, y).item() * len(x); n += len(x)
        cm += confusion(out.argmax(1), y)
    oa, iou, f1 = scores(cm)
    return (total / n if n else float("nan")), oa, iou, f1, cm


# ------------------------------------------------------------------ checks
def overfit_one_batch(model, loss_fn, dl, steps=60, lr=1e-3):
    """Sanity check on a COPY of the model: loss on one batch must drop a lot."""
    x, y = next(iter(dl)); x, y = x.to(DEV), y.to(DEV)
    tmp = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(tmp.parameters(), lr=lr)
    for s in range(steps):
        l = loss_fn(tmp(x), y); opt.zero_grad(); l.backward(); opt.step()
        if s % 10 == 0 or s == steps - 1:
            print(f"step {s:2d}  loss {l.item():.3f}")
    del tmp, opt
    if DEV == "cuda": torch.cuda.empty_cache()


ARCHS = {
    "unet_resnet34":          ("Unet", "resnet34"),
    "deeplabv3plus_resnet50": ("DeepLabV3Plus", "resnet50"),
    "segformer_mitb2":        ("Segformer", "mit_b2"),
}


def arch_of(cfg):
    """Architecture key of a run (older runs have no ARCH key -> infer from RUN_NAME)."""
    if cfg.get("ARCH"):
        return cfg["ARCH"]
    run = cfg.get("RUN_NAME", "")
    return {"A": "unet_resnet34", "B": "deeplabv3plus_resnet50", "C": "segformer_mitb2"}.get(run[:1])


def build_model(arch, weights=None, in_ch=3):
    import segmentation_models_pytorch as smp
    cls, enc = ARCHS[arch]
    return getattr(smp, cls)(encoder_name=enc, encoder_weights=weights, in_channels=in_ch, classes=NC)


def build_pretrained(arch, want=True, in_ch=3):
    """ImageNet weights if possible, otherwise random start. Returns (model, weights_used).
    With in_ch != 3 the pretrained first layer is adapted automatically by smp."""
    if want:
        try:
            return build_model(arch, "imagenet", in_ch), "imagenet"
        except Exception as e:
            print("Pretrained weights not available -> random start. Reason:", str(e)[:150])
    return build_model(arch, None, in_ch), "none"


def miou_present(cm):
    """mIoU over classes that occur in the labels or predictions of this subset only
    (absent classes are not counted as 0). Use for per-region / per-month breakdowns."""
    cm = cm.double(); tp = cm.diag(); union = cm.sum(0) + cm.sum(1) - tp
    present = union > 0
    return float((tp[present] / union[present]).mean()) if present.any() else float("nan")


def miou_label(cm):
    """mIoU over the classes that are present in the LABELS of this subset.
    Better for per-region tables: a class that is absent from the region but
    falsely predicted in a few pixels no longer counts as an IoU of 0."""
    cm = cm.double(); tp = cm.diag(); union = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    return float((tp[present] / union[present]).mean()) if present.any() else float("nan")


def oa_of(cm):
    cm = cm.double()
    return float(cm.diag().sum() / cm.sum().clamp(min=1))


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ------------------------------------------------------------------ training
def train(model, cfg, run, train_dl, val_dl, loss_fn):
    """AdamW + linear warm-up + cosine decay, mixed precision, early stopping on val mIoU.
    Resumes automatically from run/last.pth."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    warm = cfg.get("WARMUP_EPOCHS", 0)
    def lr_lambda(e):
        if e < warm:
            return (e + 1) / warm
        p = (e - warm) / max(1, cfg["EPOCHS"] - warm)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    start, history, best, bad = 0, [], -1.0, 0

    if (run / "last.pth").exists():
        ck = torch.load(run / "last.pth", map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, history, best, bad = ck["epoch"] + 1, ck["history"], ck["best"], ck["bad"]
        print(f"Resuming from epoch {start} (best val mIoU so far {best:.4f})")

    for epoch in range(start, cfg["EPOCHS"]):
        if bad >= cfg["PATIENCE"]:
            print("Early stopping."); break
        t0 = time.time(); model.train(); tl, n = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(x); n += len(x)
        sched.step()

        vl, oa, iou, f1, _ = evaluate(model, val_dl, loss_fn)
        row = dict(epoch=epoch, train_loss=tl / n, val_loss=vl, val_oa=oa,
                   val_miou=float(iou.mean()), lr=opt.param_groups[0]["lr"],
                   seconds=time.time() - t0)
        history.append(row)
        improved = row["val_miou"] > best
        if improved:
            best, bad = row["val_miou"], 0
            torch.save(model.state_dict(), run / "best.pth")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, history=history,
                        best=best, bad=bad), run / "last.pth")
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"ep {epoch:2d} | train {row['train_loss']:.3f} | val {vl:.3f} | OA {oa:.3f} | "
              f"mIoU {row['val_miou']:.3f}{'  *best*' if improved else ''} | {row['seconds']:.0f}s")
    print(f"Done. Best val mIoU: {best:.4f}")
    return pd.DataFrame(history)


# ------------------------------------------------------------------ reporting
def plot_curves(run, title):
    h = pd.read_csv(run / "history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train loss"); ax[0].plot(h.epoch, h.val_loss, label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (CE + Dice)"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.val_miou, label="val mIoU"); ax[1].plot(h.epoch, h.val_oa, label="val OA")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title(f"Validation — {title}")
    plt.tight_layout(); plt.savefig(run / "curves.png", dpi=150); plt.show()


def plot_confusion(cm, path, title):
    cmn = (cm.double() / cm.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
    plt.figure(figsize=(6.5, 5.5)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1); plt.colorbar()
    plt.xticks(range(NC), CLASSES, rotation=45, ha="right"); plt.yticks(range(NC), CLASSES)
    for i in range(NC):
        for j in range(NC):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if cmn[i, j] > 0.5 else "black")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.show()


def final_validation(model, cfg, run, val_dl):
    """Best checkpoint on validation: per-class table, confusion matrix, metrics.json."""
    model.load_state_dict(torch.load(run / "best.pth", map_location=DEV))
    _, oa, iou, f1, cm = evaluate(model, val_dl)
    res = pd.DataFrame({"IoU": iou, "F1": f1}, index=CLASSES).round(3)
    res.loc["mean"] = res.mean().round(3)
    res.to_csv(run / "val_per_class.csv")
    h = pd.read_csv(run / "history.csv")
    metrics = dict(model=cfg["MODEL_NAME"], run=cfg["RUN_NAME"], group=cfg.get("GROUP", cfg["RUN_NAME"]),
                   arch=arch_of(cfg), features=cfg.get("FEATURES", "monthly"),
                   params_M=round(count_params(model), 1),
                   val_oa=round(oa, 4), val_miou=round(float(iou.mean()), 4),
                   best_epoch=int(h.loc[h.val_miou.idxmax(), "epoch"]), epochs_run=len(h),
                   train_minutes=round(h.seconds.sum() / 60, 1),
                   per_class_iou=dict(zip(CLASSES, np.round(iou, 4).tolist())),
                   per_class_f1=dict(zip(CLASSES, np.round(f1, 4).tolist())))
    json.dump(metrics, open(run / "metrics.json", "w"), indent=2)
    print(f"{cfg['MODEL_NAME']}: val OA {oa:.3f} | val mIoU {iou.mean():.3f} | "
          f"{metrics['params_M']} M params | {metrics['train_minutes']} min")
    plot_confusion(cm, run / "val_confusion.png", f"{cfg['MODEL_NAME']} — validation (row-normalised)")
    return res


def show_predictions(model, val_dl, run, n=3, month=6):
    """Side-by-side: VV input, label, prediction for a few validation patches (July)."""
    from matplotlib.colors import ListedColormap
    from preprocess import COLORS
    cmap = ListedColormap(COLORS)
    ds = val_dl.dataset
    idx = [i for i in range(len(ds)) if ds.month[i] == month][:: max(1, (len(ds) // 12) // n)][:n]
    model.eval()
    fig, ax = plt.subplots(len(idx), 3, figsize=(12, 4 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        x, y = ds[i]
        with torch.no_grad():  # x[0] = normalised VV of that month
            p = model(x[None].to(DEV)).argmax(1)[0].cpu().numpy()
        ax[r, 0].imshow(x[0], cmap="gray"); ax[r, 0].set_title("VV (normalised)")
        ax[r, 1].imshow(np.ma.masked_equal(y.numpy(), IGNORE), cmap=cmap, vmin=0, vmax=NC - 1,
                        interpolation="nearest"); ax[r, 1].set_title("label (WorldCover)")
        ax[r, 2].imshow(p, cmap=cmap, vmin=0, vmax=NC - 1, interpolation="nearest"); ax[r, 2].set_title("prediction")
        for a in ax[r]: a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, CLASSES, loc="lower center", ncol=NC)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(run / "val_examples.png", dpi=120); plt.show()
