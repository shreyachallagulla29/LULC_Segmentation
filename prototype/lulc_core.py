"""
lulc_core.py - everything the prototype does, without the user interface.

    location (lat, lon) + start month
      -> download 12 monthly Sentinel-1 composites from Google Earth Engine   (download_s1)
      -> run the final model, SegFormer v8, on all 12 months                  (predict_12_months)
      -> class percentages per month, month-to-month and total change (pp)    (class_shares, changes)

The download and preprocessing are the same as for the training data (notebooks 01-01c, preprocess.py),
and the model code is the same as in training (train_lib_r8.py), so the prototype sees data prepared
exactly like the data the model learned from.
"""
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above prototype/)
sys.path.insert(0, str(ROOT))
from preprocess import CLASSES, COLORS, IGNORE, decode_s1, clip_db, remap_worldcover   # noqa: E402

NC = len(CLASSES)
RES = 10                                               # pixel size in metres (as in training)
RUN = ROOT / "runs" / "C_segformer_v8_border"          # the final model
WEIGHTS = RUN / "best.pth"
DATA = ROOT / "data_r5"                                # normalisation statistics of the training data
ZENODO_ZIP = "https://zenodo.org/records/24060094/files/weights_final.zip?download=1"
ZIP_MEMBER = "runs/C_segformer_v8_border/best.pth"
CACHE = Path(__file__).resolve().parent / "cache"      # downloaded radar is kept here (not in git)
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FIRST_YEAR = 2015                                      # Sentinel-1 IW data is global from 2015 on


# ------------------------------------------------------------------ months
def month_list(start_year, start_month):
    """12 (year, month) pairs starting at the start month."""
    return [((start_year * 12 + start_month - 1 + k) // 12, (start_month - 1 + k) % 12 + 1) for k in range(12)]


def month_labels(months):
    return [f"{MONTH_ABBR[m - 1]} {y}" for y, m in months]


def latest_start(today=None):
    """Latest start month whose 12 months are all complete (the last month must be over)."""
    today = today or pd.Timestamp.today()
    k = today.year * 12 + today.month - 1 - 12          # 12 complete months before the current month
    return k // 12, k % 12 + 1


# ------------------------------------------------------------------ model
def ensure_weights(progress=None):
    """Use runs/C_segformer_v8_border/best.pth if it is there; otherwise download weights_final.zip
    (0.5 GB) from Zenodo once and unpack only the final model (full precision, unchanged)."""
    if WEIGHTS.exists():
        return WEIGHTS
    import requests
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "weights_final.zip.part"
    with requests.get(ZENODO_ZIP, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                if progress and total:
                    progress(done / total, f"Downloading the model weights from Zenodo: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    with zipfile.ZipFile(tmp) as z:
        z.extract(ZIP_MEMBER, ROOT)
    tmp.unlink()
    return WEIGHTS


def load_model(device="cpu"):
    """SegFormer v8 = temporal encoder over 12 months + 18 year statistics + SegFormer MiT-B2."""
    import train_lib_r8 as r8
    cfg = json.load(open(RUN / "config.json"))
    mstats = json.load(open(DATA / "norm_stats.json"))
    ystats = json.load(open(DATA / "norm_stats_r6_yearstats.json"))
    model = r8.build_model(cfg, mstats, ystats, weights=None)      # weights=None: no ImageNet download needed
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_12_months(model, x12, tile=256, stride=128, progress=None):
    """x12: (12, 3, H, W) clipped dB, NaN = no radar  ->  pred (12, H, W) uint8 (255 = no data),
    conf (12, H, W) float (probability of the chosen class).
    The area is covered with overlapping 256 x 256 windows (as in training); overlaps are blended with a
    Hann window so no seams appear. Same method as predict_region() in train_lib_r6.py, but all 12 months
    of a window are decoded in one batch (faster on a CPU)."""
    dev = next(model.parameters()).device
    T, _, H, W = x12.shape
    Hp = int(np.ceil(max(H - tile, 0) / stride)) * stride + tile
    Wp = int(np.ceil(max(W - tile, 0) / stride)) * stride + tile
    xp = torch.full((T, 3, Hp, Wp), float("nan"))
    xp[..., :H, :W] = torch.from_numpy(x12.astype(np.float32))
    w1 = np.hanning(tile) + 1e-3
    win = torch.tensor(np.outer(w1, w1), dtype=torch.float32)
    prob = torch.zeros(T, NC, Hp, Wp); wsum = torch.zeros(Hp, Wp)
    pos = [(i, j) for i in range(0, Hp - tile + 1, stride) for j in range(0, Wp - tile + 1, stride)]
    months = torch.arange(T, device=dev)
    for n, (i, j) in enumerate(pos):
        e = model.encode(xp[None, ..., i:i + tile, j:j + tile].to(dev))
        e = {k: v.expand(T, *v.shape[1:]) for k, v in e.items()}     # the same window for all 12 months
        p = model.decode(e, months).float().softmax(1).cpu()           # (12, NC, tile, tile)
        prob[..., i:i + tile, j:j + tile] += p * win
        wsum[i:i + tile, j:j + tile] += win
        if progress:
            progress((n + 1) / len(pos), f"Running the model: window {n + 1} / {len(pos)}")
    prob = (prob / wsum)[..., :H, :W].numpy()
    pred = prob.argmax(1).astype(np.uint8); conf = prob.max(1)
    nod = np.isnan(x12).any(1)
    pred[nod] = IGNORE; conf[nod] = np.nan
    return pred, conf


# ------------------------------------------------------------------ Earth Engine download
def init_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception as e:
        raise RuntimeError(
            "Earth Engine is not ready. Open a terminal, run  earthengine authenticate  once, "
            f"and check the project ID ({project!r}). Details: {str(e)[:200]}") from e
    return ee


def _grid(ee, lat, lon, size_m):
    """Exact 10 m grid in the local UTM zone (as in training): pixel (i, j) is the same ground in every layer."""
    from rasterio.warp import transform as warp_transform
    zone = int((lon + 180) // 6) + 1
    crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    x0 = round((xs[0] - size_m / 2) / RES) * RES
    y0 = round((ys[0] + size_m / 2) / RES) * RES
    n = size_m // RES
    grid = {"dimensions": {"width": n, "height": n},
            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -RES, "translateY": y0},
            "crsCode": crs}
    aoi = ee.Geometry.Rectangle([x0, y0 - size_m, x0 + size_m, y0], proj=crs, geodesic=False)
    return crs, x0, y0, grid, aoi


def _fetch(ee, expression, grid, tries=5):
    for k in range(tries):
        try:
            return ee.data.computePixels({"expression": expression, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))                   # quota / network hiccup -> wait and retry


def _s1(ee, aoi, start, end, orbit):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit)))


def _s1_month(ee, aoi, year, month, orbit):
    """Monthly composite exactly as in training: edge mask VV < -30 dB, mean in linear power, back to dB,
    + VV-VH band, stored as dB x 100 (int16), no data = -9999."""
    start = ee.Date.fromYMD(year, month, 1)
    col = _s1(ee, aoi, start, start.advance(1, "month"), orbit)
    def to_linear(img):
        vv_vh = img.select(["VV", "VH"]).updateMask(img.select("VV").gt(-30))
        return ee.Image(10).pow(vv_vh.divide(10))
    db = col.map(to_linear).mean().log10().multiply(10)
    ratio = db.select("VV").subtract(db.select("VH")).rename("VV_VH")
    return db.addBands(ratio).select(["VV", "VH", "VV_VH"]).multiply(100).round().unmask(-9999).toInt16()


def _parallel(fn, text, progress=None):
    """fn(0..11) on 4 threads; progress is reported from the calling thread (needed by Streamlit)."""
    out = [None] * 12
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn, k): k for k in range(12)}
        for n, f in enumerate(as_completed(futures)):
            out[futures[f]] = f.result()
            if progress:
                progress((n + 1) / 12, f"{text}: month {n + 1} / 12")
    return out


def download_s1(project, lat, lon, months, size_km, progress=None):
    """12 monthly Sentinel-1 composites for a size_km x size_km square around (lat, lon).
    Returns dict(x12=(12, 3, H, W) clipped dB with NaN, crs, transform, orbit, images=[n per month]).
    Results are cached in prototype/cache/, so the same request is not downloaded twice."""
    from rasterio.transform import from_origin
    size_m = int(round(size_km * 1000 / RES)) * RES
    CACHE.mkdir(exist_ok=True)
    key = f"s1_{lat:.4f}_{lon:.4f}_{months[0][0]}-{months[0][1]:02d}_{size_m}m.npz"
    if (CACHE / key).exists():
        d = np.load(CACHE / key, allow_pickle=True)
        raw, meta = d["raw"], d["meta"].item()
    else:
        ee = init_ee(project)
        crs, x0, y0, grid, aoi = _grid(ee, lat, lon, size_m)
        t0 = ee.Date.fromYMD(*months[0], 1); t1 = ee.Date.fromYMD(*months[-1], 1).advance(1, "month")
        counts = {}
        for p in ("ASCENDING", "DESCENDING"):                  # one orbit direction for the whole year
            ts = pd.to_datetime(_s1(ee, aoi, t0, t1, p).aggregate_array("system:time_start").getInfo(), unit="ms")
            counts[p] = pd.Series(list(zip(ts.year, ts.month))).value_counts().to_dict() if len(ts) else {}
        orbit = max(counts, key=lambda p: (sum(counts[p].get(ym, 0) >= 2 for ym in months), sum(counts[p].values())))
        images = [int(counts[orbit].get(ym, 0)) for ym in months]
        if sum(images) == 0:
            raise RuntimeError("No Sentinel-1 images were found for this place and period.")
        n = size_m // RES
        def one_month(k):
            if images[k] == 0:
                return np.full((3, n, n), -9999, np.int16)
            a = _fetch(ee, _s1_month(ee, aoi, *months[k], orbit), grid)
            return np.stack([a["VV"], a["VH"], a["VV_VH"]]).astype(np.int16)
        raw = np.stack(_parallel(one_month, "Downloading Sentinel-1 from Earth Engine", progress))   # (12, 3, n, n)
        meta = dict(crs=crs, transform=tuple(from_origin(x0, y0, RES, RES))[:6], orbit=orbit, images=images)
        np.savez_compressed(CACHE / key, raw=raw, meta=np.array(meta, dtype=object))
    x12 = clip_db(decode_s1(raw))                                     # same decoding / clipping as training
    return dict(x12=x12, **meta)


def download_reference(project, lat, lon, size_km):
    """ESA WorldCover 2021 for the same area, remapped to our 7 classes (255 = no data).
    For reference only - not used by the model."""
    size_m = int(round(size_km * 1000 / RES)) * RES
    CACHE.mkdir(exist_ok=True)
    key = f"worldcover_{lat:.4f}_{lon:.4f}_{size_m}m.npz"
    if (CACHE / key).exists():
        return dict(worldcover=np.load(CACHE / key)["wc"])
    ee = init_ee(project)
    crs, x0, y0, grid, aoi = _grid(ee, lat, lon, size_m)
    wc = remap_worldcover(_fetch(ee, ee.ImageCollection("ESA/WorldCover/v200").first().select("Map"), grid)["Map"])
    np.savez_compressed(CACHE / key, wc=wc)
    return dict(worldcover=wc)


# ------------------------------------------------------------------ results
def class_shares(pred, labels):
    """% of the area (pixels with radar in that month) covered by each class, per month."""
    rows = []
    for k in range(len(pred)):
        v = pred[k] != IGNORE
        rows.append([(pred[k][v] == c).mean() * 100 if v.any() else np.nan for c in range(NC)])
    return pd.DataFrame(rows, index=labels, columns=CLASSES)


def changes(shares, area_km2):
    """Month-to-month change and total change (first -> last month), in percentage points (pp).
    Example: 20 % -> 23 % = +3 pp."""
    mom = shares.diff().iloc[1:]
    mom.index = [f"{a} → {b}" for a, b in zip(shares.index[:-1], shares.index[1:])]
    total = pd.DataFrame({f"{shares.index[0]} (%)": shares.iloc[0], f"{shares.index[-1]} (%)": shares.iloc[-1],
                          "change (pp)": shares.iloc[-1] - shares.iloc[0]})
    total["change (km²)"] = total["change (pp)"] / 100 * area_km2
    return mom, total


def stability(pred, conf, labels):
    """Quality indicators per month: radar coverage, mean model confidence, and % of pixels whose class
    differs from the previous month (real land cover changes little within a year, so most flips are noise)."""
    rows = []
    for k in range(len(pred)):
        v = pred[k] != IGNORE
        flip = np.nan
        if k and (v & (pred[k - 1] != IGNORE)).any():
            both = v & (pred[k - 1] != IGNORE)
            flip = (pred[k][both] != pred[k - 1][both]).mean() * 100
        rows.append({"radar coverage (%)": v.mean() * 100,
                     "mean confidence": np.nanmean(conf[k]) if v.any() else np.nan,
                     "pixels changed vs previous month (%)": flip})
    return pd.DataFrame(rows, index=labels)


def agreement(pred, ref):
    """Agreement of one predicted map with a reference map: OA and mIoU over classes covering >= 1 % of the reference."""
    ok = (pred != IGNORE) & (ref != IGNORE)
    if not ok.any():
        return dict(OA=np.nan, mIoU=np.nan)
    cm = np.bincount(ref[ok].astype(np.int64) * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    present = cm.sum(1) >= 0.01 * cm.sum()
    return dict(OA=tp.sum() / cm.sum(), mIoU=float(np.nanmean(iou[present])))


def overlap_with_known_regions(lat, lon, size_km):
    """Does the square overlap one of the 52 regions used for training / validation / test of v8?"""
    regions = pd.read_csv(ROOT / "data" / "regions_r5.csv")
    out = []
    for r in regions.itertuples():
        dx = (r.lon - lon) * 111.32 * np.cos(np.radians(lat)); dy = (r.lat - lat) * 110.57
        ox = max(0.0, min(size_km / 2, dx + 6) - max(-size_km / 2, dx - 6))
        oy = max(0.0, min(size_km / 2, dy + 6) - max(-size_km / 2, dy - 6))
        if ox * oy > 0:
            out.append((r.name, r.split, ox * oy / size_km ** 2))
    return sorted(out, key=lambda o: -o[2])


def to_geotiff(pred, crs, transform, labels):
    """12-band GeoTIFF (one band per month, values 0-6 = classes, 255 = no data) as bytes."""
    import rasterio
    from rasterio.io import MemoryFile
    from affine import Affine
    T, H, W = pred.shape
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=W, height=H, count=T, dtype="uint8", nodata=IGNORE,
                      crs=crs, transform=Affine(*transform), compress="deflate") as dst:
            dst.write(pred)
            for k, lab in enumerate(labels):
                dst.set_band_description(k + 1, lab)
            dst.update_tags(classes=", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)), model="SegFormer v8")
            dst.write_colormap(1, {i: tuple(int(c[j:j + 2], 16) for j in (1, 3, 5)) + (255,)
                                   for i, c in enumerate(COLORS)})
        return mem.read()
