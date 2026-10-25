"""
preprocess.py  -  shared preprocessing for the Sentinel-1 LULC project.

Used by 02_preprocessing.ipynb now, and later by training and the prototype,
so that the model always sees data prepared in exactly the same way.
"""
import json
import numpy as np

# ---------------------------------------------------------------- classes
CLASSES = ["Trees", "Grass/Shrub", "Cropland", "Built-up", "Bare", "Water", "Wetland"]
IGNORE = 255
COLORS = ["#1a7f37", "#b5d36b", "#f2c94c", "#d73a49", "#a0896b", "#2f80ed", "#56ccf2"]

# ESA WorldCover code -> our class id (anything not listed -> 255 = ignore)
WORLDCOVER_TO_CLASS = {
    10: 0,              # Tree cover            -> Trees
    20: 1, 30: 1, 100: 1,  # Shrub, Grass, Moss  -> Grass/Shrub
    40: 2,              # Cropland              -> Cropland
    50: 3,              # Built-up              -> Built-up
    60: 4,              # Bare / sparse veg.    -> Bare
    80: 5,              # Permanent water       -> Water
    90: 6, 95: 6,       # Herb. wetland, Mangrove -> Wetland
    # 70 Snow & ice and 0 no-data -> ignore
}

_LUT = np.full(256, IGNORE, dtype=np.uint8)
for code, cls in WORLDCOVER_TO_CLASS.items():
    _LUT[code] = cls


def remap_worldcover(label):
    """WorldCover codes (uint8) -> class ids 0..6, 255 = ignore."""
    return _LUT[label.astype(np.uint8)]


# ---------------------------------------------------------------- radar values
BANDS = ["VV", "VH", "VV_VH"]          # VV_VH = VV - VH in dB (the VV/VH ratio)
NODATA = -9999                         # value used in the downloaded GeoTIFFs
SCALE = 100.0                          # GeoTIFFs store dB x 100 as int16
CLIP = {"VV": (-25.0, 5.0), "VH": (-32.0, -5.0), "VV_VH": (0.0, 20.0)}


def decode_s1(raw):
    """int16 (dB x 100, -9999 = no data) -> float32 dB, no data -> NaN.
    Works for any array whose 3rd-last axis is the band axis (..., 3, H, W)."""
    x = raw.astype(np.float32)
    x[raw == NODATA] = np.nan
    return x / SCALE


def clip_db(x):
    """Clip each band to a sensible dB range. NaN stays NaN.
    x: (..., 3, H, W) in dB."""
    x = x.copy()
    for b, name in enumerate(BANDS):
        lo, hi = CLIP[name]
        x[..., b, :, :] = np.clip(x[..., b, :, :], lo, hi)
    return x


def normalise(x, stats):
    """(x - mean) / std per band, NaN -> 0 (= the band mean).
    x: (..., 3, H, W) clipped dB; stats: dict loaded from norm_stats.json."""
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    return np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)


def load_stats(path="norm_stats.json"):
    with open(path) as f:
        return json.load(f)
