# Sentinel-1 Land-Cover Mapping (radar only)

Semantic segmentation of **7 land-cover classes** (Trees, Grass/Shrub, Cropland, Built-up, Bare, Water, Wetland)
from **Sentinel-1 radar (VV, VH)** at 10 m, with **ESA WorldCover 2021** as labels.

## Status: round 2 — model comparison (U-Net, DeepLabV3+, SegFormer)

- Three networks trained with **identical** data, augmentation, loss and training loop (`train_lib.py`):
  U-Net / ResNet-34 (04a), DeepLabV3+ / ResNet-50 (04b), SegFormer / MiT-B2 (04c).
- Notebook 05 compares them on the validation regions and on the 6 test regions.

## Results so far

| Round | Run | What changed | Val mIoU | Test mIoU |
|---|---|---|---|---|
| 1-2 | `A_round2` (U-Net / ResNet-34) | first model, 4 val regions | 0.508 | — |
| 2 | A / B / C (v1) | U-Net vs DeepLabV3+ vs SegFormer | 0.430 / 0.497 / 0.502 | 0.577 / 0.605 / 0.534 |

Scores of round 2 on its own 4 validation regions: U-Net 0.496, DeepLabV3+ 0.493, SegFormer 0.515 (`runs/comparison_val.csv`).

Run names: **A** = U-Net, **B** = DeepLabV3+, **C** = SegFormer; v1 (no number) … v2 = improvement rounds; "b" = alternative idea.

## Folder structure

```
├── preprocess.py           shared data rules (decoding, clipping, 7 classes, colours, normalisation)
├── train_lib.py            training code, round 2  (notebooks 04a-c, 05)
├── notebooks/
│   ├── r1_data_first_model/      01 download, 02 preprocessing, 03 first U-Net
│   ├── r2_model_comparison/      04a U-Net, 04b DeepLabV3+, 04c SegFormer, 05 compare
├── data/                   region lists, class statistics  (radar + labels: see "Data")
└── runs/                   one folder per trained model: config, metrics, curves, confusion matrices
                            (weights *.pth: see "Model weights")
```

## Data

The data is **not in git**. It is archived on Zenodo: **[doi.org/10.5281/zenodo.24060094](https://doi.org/10.5281/zenodo.24060094)**.
Details per round in [`data/README.md`](data/README.md).

1. **Download** the packages this round needs and unzip each **in the project folder** (the paths inside the zips put every file in place):

   | Package | Size | Needed for |
   |---|---|---|
   | [`tiles_r1-2.zip`](https://zenodo.org/records/24060094/files/tiles_r1-2.zip?download=1) | 3.0 GB | first U-Net, v1 |
   | [`raw_geotiffs.zip`](https://zenodo.org/records/24060094/files/raw_geotiffs.zip?download=1) | 5.4 GB | rebuilding the tiles without Earth Engine |

   Check a download with [`SHA256SUMS.txt`](https://zenodo.org/records/24060094/files/SHA256SUMS.txt?download=1) (PowerShell: `Get-FileHash file.zip`).
2. **Or recreate** it (free Google Earth Engine account; set your project in the download notebooks):
   run the download notebooks, then the preprocessing notebooks, in round order (see `data/README.md`).

Files that **are** in git because the code needs them: `data/regions.csv`, `data/patch_index.csv`, `data/norm_stats.json`.

## Model weights

`best.pth` files (≈ 95 MB each) are **not in git**. [`weights_all.zip`](https://zenodo.org/records/24060094/files/weights_all.zip?download=1) (1.8 GB) contains
the weights of all runs; unzip it in the project folder (each file lands in its `runs/<run name>/` folder).
Runs up to this round: `A_round2`, `A_unet_resnet34`, `B_deeplabv3plus_resnet50`, `C_segformer_mitb2`.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

All notebooks start with a cell that switches to the project folder, so they can be run from their sub-folder.
