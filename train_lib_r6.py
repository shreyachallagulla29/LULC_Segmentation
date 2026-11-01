"""
train_lib_r6.py  -  round 6: the model sees ALL 12 months at once + richer year statistics.

What is new compared with round 5 (train_lib_r5.py, which stays unchanged):

  Step 2 - richer year statistics (18 channels, computed per pixel from the 12 months):
           for VV, VH and VV-VH: median, 10th percentile, 90th percentile, minimum, maximum, range.
  Step 4 - the whole 12-month series goes into the network. A small TEMPORAL ENCODER looks along
           the time axis of every pixel (1-D convolutions over the months, the same for every
           pixel) and produces
             * a feature vector for EVERY month  (what happens in this month, seen in context), and
             * a summary of the whole year      (average and maximum of the monthly features).
           For the map of month m, the features of month m + the year summary + the 18 year
           statistics + the raw values of month m are merged (1x1 convolution) and passed to the
           same SegFormer (MiT-B2, ImageNet-pretrained) as before.
           -> the map can differ between months (month features), but is still guided by the year.

The year statistics are computed INSIDE the model from the raw clipped dB values, so training,
evaluation, notebook 15 and the prototype can never compute them differently.

Input of the model: x (B, 12, 3, H, W) clipped dB, NaN = no radar; m (B,) month index 0..11.
"""
import json, random, time, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from preprocess import CLASSES, COLORS, IGNORE, load_stats
from train_lib_r5 import make_loss, scores, confusion, plot_confusion, count_params, miou_label, oa_of

NC = len(CLASSES)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STAT_NAMES = [f"{b}_{s}" for b in ["VV", "VH", "VV_VH"] for s in ["median", "p10", "p90", "min", "max", "range"]]


# ------------------------------------------------------------------ step 2: year statistics
def year_stats_torch(x):
    """x: (B, T, 3, H, W) dB with NaN -> (B, 18, H, W); NaN where a pixel has no data at all.
    Per band: median, 10th and 90th percentile (linear interpolation, like numpy), min, max, range."""
    x = x.float()
    nan = torch.isnan(x)
    s = torch.sort(torch.where(nan, torch.full_like(x, float("inf")), x), dim=1).values   # NaN last
    n = (~nan).sum(1, keepdim=True)                                                          # valid months
    def q(p):
        pos = (n - 1).clamp(min=0).float() * p
        lo, hi = pos.floor().long(), pos.ceil().long()
        vlo, vhi = s.gather(1, lo), s.gather(1, hi)
        return (vlo + (vhi - vlo) * (pos - lo)).squeeze(1)
    mn = s[:, 0]; mx = s.gather(1, (n - 1).clamp(min=0)).squeeze(1)
    out = torch.stack([q(0.5), q(0.1), q(0.9), mn, mx, mx - mn], 2)          # (B, 3, 6, H, W)
    out = out.reshape(x.shape[0], 18, *x.shape[-2:])
    nodata = (n == 0).squeeze(1).repeat_interleave(6, 1)                     # (B, 18, H, W)
    return torch.where(nodata | ~torch.isfinite(out), torch.full_like(out, float("nan")), out)


def year_stat_norm(index, data, path=None, max_patches=300):
    """Mean / std of the 18 year statistics over TRAINING patches (cached as JSON)."""
    path = Path(path or Path(data) / "norm_stats_r6_yearstats.json")
    if path.exists():
        return json.load(open(path))
    files = index[index.split == "train"].file.tolist()
    random.Random(0).shuffle(files)
    n = torch.zeros(18, dtype=torch.float64); s = torch.zeros(18, dtype=torch.float64); ss = torch.zeros_like(s)
    for f in tqdm(files[:max_patches], desc="year-statistics normalisation"):
        x = torch.from_numpy(np.load(Path(data) / f)["x"].astype(np.float32))[None].to(DEV)
        st = year_stats_torch(x)[0].reshape(18, -1).double().cpu()
        ok = ~torch.isnan(st)
        n += ok.sum(1); s += torch.where(ok, st, 0).sum(1); ss += torch.where(ok, st ** 2, 0).sum(1)
    mean = s / n; std = (ss / n - mean ** 2).sqrt()
    out = {"names": STAT_NAMES, "mean": mean.round(decimals=4).tolist(), "std": std.round(decimals=4).tolist()}
    json.dump(out, open(path, "w"), indent=2)
    return out


# ------------------------------------------------------------------ step 4: the model
class TemporalSegFormer(nn.Module):
    """Temporal encoder over the 12 months (per pixel) + year statistics -> SegFormer."""
    def __init__(self, mstats, ystats, temp_dim=24, fuse_dim=32, weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.register_buffer("m_mean", torch.tensor(mstats["mean"], dtype=torch.float32).view(1, 1, 3, 1, 1))
        self.register_buffer("m_std", torch.tensor(mstats["std"], dtype=torch.float32).view(1, 1, 3, 1, 1))
        self.register_buffer("y_mean", torch.tensor(ystats["mean"], dtype=torch.float32).view(1, 18, 1, 1))
        self.register_buffer("y_std", torch.tensor(ystats["std"], dtype=torch.float32).view(1, 18, 1, 1))
        d = temp_dim
        # 1-D convolutions along the month axis only (kernel 3 months x 1 x 1 pixel);
        # circular padding: December is next to January, works for any start month
        self.temporal = nn.Sequential(
            nn.Conv3d(4, d, (3, 1, 1), padding=(1, 0, 0), padding_mode="circular"), nn.BatchNorm3d(d), nn.GELU(),
            nn.Conv3d(d, d, (3, 1, 1), padding=(1, 0, 0), padding_mode="circular"), nn.BatchNorm3d(d), nn.GELU(),
            nn.Conv3d(d, d, (3, 1, 1), padding=(1, 0, 0), padding_mode="circular"), nn.BatchNorm3d(d), nn.GELU())
        cin = 3 * d + 18 + 3 + 1
        self.fuse = nn.Sequential(nn.Conv2d(cin, fuse_dim, 1), nn.BatchNorm2d(fuse_dim), nn.GELU())
        self.seg = smp.Segformer(encoder_name="mit_b2", encoder_weights=weights, in_channels=fuse_dim, classes=NC)

    def encode(self, x):
        """Everything that is the same for all months: temporal features, year summary, year stats."""
        with torch.autocast(device_type="cuda", enabled=False):
            x = x.float()
            valid = (~torch.isnan(x).any(2)).float()                        # (B, T, H, W)
            ys = (year_stats_torch(x) - self.y_mean) / self.y_std
            ys = torch.nan_to_num(ys, nan=0.0)
            xn = torch.nan_to_num((x - self.m_mean) / self.m_std, nan=0.0)  # (B, T, 3, H, W)
        inp = torch.cat([xn, valid[:, :, None]], 2).permute(0, 2, 1, 3, 4)   # (B, 4, T, H, W)
        f = self.temporal(inp) * valid[:, None]                              # (B, d, T, H, W)
        cnt = valid.sum(1, keepdim=True).clamp(min=1)[:, None]               # (B, 1, 1, H, W)
        f_mean = f.sum(2, keepdim=True) / cnt
        f_max = torch.where(valid[:, None] > 0, f, torch.full_like(f, -1e4)).amax(2, keepdim=True)
        f_max = torch.where(f_max < -1e3, torch.zeros_like(f_max), f_max)
        return dict(xn=xn, valid=valid, ys=ys, f=f, summary=torch.cat([f_mean, f_max], 1)[:, :, 0])

    def decode(self, e, m):
        """Map for month m (tensor (B,) of month indices) from the encoded series."""
        b = torch.arange(len(m), device=m.device)
        z = torch.cat([e["f"][b, :, m], e["summary"], e["ys"].to(e["f"].dtype),
                       e["xn"][b, m].to(e["f"].dtype), e["valid"][b, m][:, None].to(e["f"].dtype)], 1)
        return self.seg(self.fuse(z))

    def forward(self, x, m):
        return self.decode(self.encode(x), m)


def build_model(cfg, mstats, ystats, weights="imagenet"):
    return TemporalSegFormer(mstats, ystats, cfg.get("TEMP_DIM", 24), cfg.get("FUSE_DIM", 32), weights)


# ------------------------------------------------------------------ data
class TrainSeries(Dataset):
    """Whole 12-month series per patch. Augmentation: flips / 90° rotations, one ±NOISE_DB dB
    calibration offset per band (same for all months), random circular shift of the months
    (= a different start month, as in the prototype) and random dropping of months (missing data)."""
    def __init__(self, files, data, noise_db=2.0, roll=True, month_drop=0.1):
        self.files, self.data = list(files), Path(data)
        self.noise_db, self.roll, self.month_drop = noise_db, roll, month_drop
    def __len__(self): return len(self.files)
    def __getitem__(self, k):
        d = np.load(self.data / self.files[k])
        x, y = d["x"].astype(np.float32), d["y"].astype(np.int64)
        x = x + np.random.uniform(-self.noise_db, self.noise_db, (1, 3, 1, 1)).astype(np.float32)
        if self.roll:
            x = np.roll(x, random.randrange(12), axis=0)
        m = random.randrange(12)
        if self.month_drop:
            drop = np.random.rand(12) < self.month_drop; drop[m] = False
            x[drop] = np.nan
        y = np.where(np.isnan(x[m]).any(0), IGNORE, y)                     # no radar in month m -> ignore
        if random.random() < 0.5: x, y = x[..., ::-1], y[:, ::-1]
        if random.random() < 0.5: x, y = x[..., ::-1, :], y[::-1]
        r = random.randrange(4); x, y = np.rot90(x, r, (2, 3)), np.rot90(y, r)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(m),
                torch.from_numpy(np.ascontiguousarray(y)))


class EvalSeries(Dataset):
    """Validation / test: one item = one patch with all 12 months (scored for every month)."""
    def __init__(self, files, data, desc="loading"):
        self.xs, self.ys, self.files = [], [], list(files)
        for f in tqdm(self.files, desc=desc):
            d = np.load(Path(data) / f)
            self.xs.append(torch.from_numpy(d["x"].astype(np.float16)))
            self.ys.append(torch.from_numpy(d["y"].astype(np.int64)))
    def __len__(self): return len(self.xs)
    def __getitem__(self, k): return self.xs[k].float(), self.ys[k]


def make_loaders(cfg, data, index, val_ds=None):
    train_ds = TrainSeries(index[index.split == "train"].file, data, cfg.get("NOISE_DB", 2.0),
                           cfg.get("ROLL", True), cfg.get("MONTH_DROP", 0.1))
    val_ds = val_ds or EvalSeries(index[index.split == "val"].file, data, "loading val")
    train_dl = DataLoader(train_ds, cfg["BATCH_SIZE"], shuffle=True, drop_last=True,
                          num_workers=cfg["NUM_WORKERS"], pin_memory=(DEV == "cuda"))
    val_dl = DataLoader(val_ds, cfg.get("EVAL_BATCH", 2), shuffle=False)
    print(f"train patches {len(train_ds)} | val patches {len(val_ds)} (x 12 months each)")
    return train_dl, val_dl


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, dl, loss_fn=None, per_month=False):
    """Scores every month of every patch. Returns loss, OA, IoU, F1, confusion (+ per-month confusions)."""
    model.eval(); total, n = 0.0, 0
    cm = torch.zeros(NC, NC, dtype=torch.long, device=DEV)
    cms = torch.zeros(12, NC, NC, dtype=torch.long, device=DEV)
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            e = model.encode(x)
            for m in range(12):
                mm = torch.full((len(x),), m, device=DEV, dtype=torch.long)
                out = model.decode(e, mm)
                ym = torch.where(torch.isnan(x[:, m]).any(1), torch.full_like(y, IGNORE), y)
                if loss_fn is not None:
                    total += loss_fn(out, ym).item() * len(x); n += len(x)
                c = confusion(out.argmax(1), ym); cm += c; cms[m] += c
    oa, iou, f1 = scores(cm)
    res = ((total / n if n else float("nan")), oa, iou, f1, cm)
    return res + (cms,) if per_month else res


# ------------------------------------------------------------------ training
def train(model, cfg, run, train_dl, val_dl, loss_fn):
    """AdamW (SegFormer: LR, new layers: LR_NEW) + warm-up + cosine, mixed precision,
    early stopping on val mIoU, resumes from run/last.pth."""
    new = [p for n_, p in model.named_parameters() if not n_.startswith("seg.encoder")]
    enc = [p for n_, p in model.named_parameters() if n_.startswith("seg.encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["LR"]}, {"params": new, "lr": cfg["LR_NEW"]}],
                            weight_decay=cfg["WEIGHT_DECAY"])
    warm = cfg.get("WARMUP_EPOCHS", 0)
    def lr_lambda(e):
        if e < warm: return (e + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, cfg["EPOCHS"] - warm)))
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
        for x, m, y in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            x, m, y = x.to(DEV, non_blocking=True), m.to(DEV), y.to(DEV, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                out = model(x, m)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(x); n += len(x)
        sched.step()
        vl, oa, iou, f1, _ = evaluate(model, val_dl, loss_fn)
        row = dict(epoch=epoch, train_loss=tl / n, val_loss=vl, val_oa=oa, val_miou=float(iou.mean()),
                   lr=opt.param_groups[0]["lr"], seconds=time.time() - t0)
        history.append(row)
        improved = row["val_miou"] > best
        if improved:
            best, bad = row["val_miou"], 0
            torch.save(model.state_dict(), run / "best.pth")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, history=history, best=best, bad=bad),
                   run / "last.pth")
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"ep {epoch:2d} | train {row['train_loss']:.3f} | val {vl:.3f} | OA {oa:.3f} | "
              f"mIoU {row['val_miou']:.3f}{'  *best*' if improved else ''} | {row['seconds']:.0f}s")
    print(f"Done. Best val mIoU: {best:.4f}")
    return pd.DataFrame(history)


# ------------------------------------------------------------------ reporting
def plot_curves(run, title, loss_name="CE + Lovász"):
    h = pd.read_csv(run / "history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train loss"); ax[0].plot(h.epoch, h.val_loss, label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(f"loss ({loss_name})"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.val_miou, label="val mIoU"); ax[1].plot(h.epoch, h.val_oa, label="val OA")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title(f"Validation — {title}")
    plt.tight_layout(); plt.savefig(run / "curves.png", dpi=150); plt.show()


def report(model, cfg, run, dl, split="val"):
    """Best checkpoint on a split: per-class table, confusion matrix, per-month scores, metrics json."""
    _, oa, iou, f1, cm, cms = evaluate(model, dl, per_month=True)
    res = pd.DataFrame({"IoU": iou, "F1": f1}, index=CLASSES).round(3)
    res.loc["mean"] = res.mean().round(3)
    res.to_csv(run / f"{split}_per_class.csv")
    pm = pd.Series([float(scores(c)[1].mean()) for c in cms], index=[f"{m + 1:02d}" for m in range(12)])
    h = pd.read_csv(run / "history.csv")
    metrics = dict(model=cfg["MODEL_NAME"], run=cfg["RUN_NAME"], arch=cfg["ARCH"], features=cfg["FEATURES"],
                   params_M=round(count_params(model), 1), **{f"{split}_oa": round(oa, 4),
                   f"{split}_miou": round(float(iou.mean()), 4), f"{split}_month_spread": round(pm.max() - pm.min(), 4)},
                   best_epoch=int(h.loc[h.val_miou.idxmax(), "epoch"]), epochs_run=len(h),
                   train_minutes=round(h.seconds.sum() / 60, 1),
                   per_class_iou=dict(zip(CLASSES, np.round(iou, 4).tolist())),
                   per_class_f1=dict(zip(CLASSES, np.round(f1, 4).tolist())))
    old = json.load(open(run / "metrics.json")) if (run / "metrics.json").exists() else {}
    old.update({k: v for k, v in metrics.items() if split == "val" or k.startswith(split)})
    if split != "val":
        old[f"{split}_per_class_iou"] = metrics["per_class_iou"]
    json.dump(old, open(run / "metrics.json", "w"), indent=2)
    print(f"{cfg['MODEL_NAME']} - {split}: OA {oa:.3f} | mIoU {iou.mean():.3f} | "
          f"best / worst month mIoU {pm.max():.3f} / {pm.min():.3f} (spread {pm.max() - pm.min():.3f})")
    plot_confusion(cm, run / f"{split}_confusion.png", f"{cfg['MODEL_NAME']} — {split} (row-normalised)")
    return res, pm


# ------------------------------------------------------------------ whole region (notebook 15 / prototype)
@torch.no_grad()
def predict_region(model, x12, tile=256, stride=128):
    """x12: (12, 3, H, W) clipped dB with NaN (numpy) -> pred (12, H, W) uint8 (255 = no data), conf (12, H, W).
    Sliding window with a Hann weighting, as in notebook 15."""
    model.eval()
    T, _, H, W = x12.shape
    Hp = int(np.ceil(max(H - tile, 0) / stride)) * stride + tile
    Wp = int(np.ceil(max(W - tile, 0) / stride)) * stride + tile
    xp = torch.full((T, 3, Hp, Wp), float("nan")); xp[..., :H, :W] = torch.from_numpy(x12.astype(np.float32))
    w1 = np.hanning(tile) + 1e-3; win = torch.tensor(np.outer(w1, w1), dtype=torch.float32, device=DEV)
    prob = torch.zeros(T, NC, Hp, Wp); wsum = torch.zeros(Hp, Wp)
    for i in range(0, Hp - tile + 1, stride):
        for j in range(0, Wp - tile + 1, stride):
            xt = xp[None, ..., i:i + tile, j:j + tile].to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                e = model.encode(xt)
                for m in range(T):
                    p = model.decode(e, torch.tensor([m], device=DEV)).float().softmax(1)[0]
                    prob[m, :, i:i + tile, j:j + tile] += (p * win).cpu()
            wsum[i:i + tile, j:j + tile] += win.cpu()
    prob = (prob / wsum)[..., :H, :W].numpy()
    pred = prob.argmax(1).astype(np.uint8); conf = prob.max(1)
    nod = np.isnan(x12).any(1)
    pred[nod] = IGNORE; conf[nod] = np.nan
    return pred, conf
