"""
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


class TrainPatches(Dataset):
    """One random month per patch per epoch + flips / 90° rotations / ±1 dB offset."""
    def __init__(self, files, data, stats):
        self.files, self.data = list(files), data
        self.mean = np.array(stats["mean"], np.float32)[:, None, None]
        self.std = np.array(stats["std"], np.float32)[:, None, None]
    def __len__(self): return len(self.files)
    def __getitem__(self, k):
        d = np.load(self.data / self.files[k])
        m = random.randrange(12)
        x, y = d["x"][m].astype(np.float32), d["y"]
        x = x + np.random.uniform(-1, 1, (3, 1, 1)).astype(np.float32)
        if random.random() < 0.5: x, y = x[:, :, ::-1], y[:, ::-1]
        if random.random() < 0.5: x, y = x[:, ::-1], y[::-1]
        r = random.randrange(4); x, y = np.rot90(x, r, (1, 2)), np.rot90(y, r)
        x, y = _prep(np.ascontiguousarray(x), np.ascontiguousarray(y), self.mean, self.std)
        return torch.from_numpy(x), torch.from_numpy(y)


class EvalPatches(Dataset):
    """All patches x all 12 months, no augmentation, preloaded (half precision)."""
    def __init__(self, files, data, stats, desc="loading"):
        mean = np.array(stats["mean"], np.float32)[:, None, None]
        std = np.array(stats["std"], np.float32)[:, None, None]
        self.xs, self.ys, self.month = [], [], []
        for f in tqdm(list(files), desc=desc):
            d = np.load(data / f)
            for m in range(12):
                x, y = _prep(d["x"][m].astype(np.float32), d["y"], mean, std)
                self.xs.append(torch.from_numpy(x).half())
                self.ys.append(torch.from_numpy(y).to(torch.uint8))
                self.month.append(m)
    def __len__(self): return len(self.xs)
    def __getitem__(self, k): return self.xs[k].float(), self.ys[k].long()


def make_loaders(cfg, data, stats, index):
    train_ds = TrainPatches(index[index.split == "train"].file, data, stats)
    val_ds = EvalPatches(index[index.split == "val"].file, data, stats, "loading val")
    train_dl = DataLoader(train_ds, cfg["BATCH_SIZE"], shuffle=True, drop_last=True,
                          num_workers=cfg["NUM_WORKERS"], pin_memory=(DEV == "cuda"),
                          persistent_workers=cfg["NUM_WORKERS"] > 0)
    val_dl = DataLoader(val_ds, 32, shuffle=False)
    print(f"train patches {len(train_ds)} | val samples {len(val_ds)} "
          f"({len(val_ds) // 12} patches x 12 months)")
    return train_dl, val_dl


# ------------------------------------------------------------------ loss + metrics
def make_loss(cfg, stats):
    import segmentation_models_pytorch as smp
    cw = (torch.tensor(stats["class_weights"], dtype=torch.float32, device=DEV)
          if cfg["USE_CLASS_WEIGHTS"] else None)
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=IGNORE)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    def loss_fn(out, y):
        out = out.float()
        return ce(out, y) + (dice(out, y) if cfg["USE_DICE"] else 0.0)
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
    metrics = dict(model=cfg["MODEL_NAME"], run=cfg["RUN_NAME"], params_M=round(count_params(model), 1),
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
        with torch.no_grad():
            p = model(x[None].to(DEV)).argmax(1)[0].cpu().numpy()
        ax[r, 0].imshow(x[0], cmap="gray"); ax[r, 0].set_title("VV (normalised)")
        ax[r, 1].imshow(np.ma.masked_equal(y.numpy(), IGNORE), cmap=cmap, vmin=0, vmax=NC - 1,
                        interpolation="nearest"); ax[r, 1].set_title("label (WorldCover)")
        ax[r, 2].imshow(p, cmap=cmap, vmin=0, vmax=NC - 1, interpolation="nearest"); ax[r, 2].set_title("prediction")
        for a in ax[r]: a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, CLASSES, loc="lower center", ncol=NC)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(run / "val_examples.png", dpi=120); plt.show()
