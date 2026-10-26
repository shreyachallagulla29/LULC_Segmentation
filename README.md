# Sentinel-1 Land-Cover Mapping (radar only)

Semantic segmentation of **7 land-cover classes** (Trees, Grass/Shrub, Cropland, Built-up, Bare, Water, Wetland)
from **Sentinel-1 radar (VV, VH)** at 10 m, with **ESA WorldCover 2021** as labels.

## Status: round 5 — 11 targeted training regions (v5, v5b)

- **11 new training regions** chosen for the weakest classes (download 01c, preprocessing 02c → `data_r5`).
- **v5:** SegFormer v4 trained on the larger dataset; **v5b:** v4b on the larger dataset.
- Notebook 05d compares everything on the same val / test regions.

## Results so far
(rounds 2+ re-scored on the same 8 validation and 6 test regions)

| Round | Run | What changed | Val mIoU | Test mIoU |
|---|---|---|---|---|
| 1-2 | `A_round2` (U-Net / ResNet-34) | first model, 4 val regions | 0.508 | — |
| 2 | A / B / C (v1) | U-Net vs DeepLabV3+ vs SegFormer | 0.430 / 0.497 / 0.502 | 0.577 / 0.605 / 0.534 |
| 3 | A / B / C v2 | 8 val regions, Lovász loss, longer training | 0.488 / 0.500 / 0.536 | 0.625 / 0.610 / 0.616 |
| 3 | C v3 | v2 + regularisation | 0.543 | 0.618 |
| 4 | A / C v4 | + year context (12-month mean / std of VV, VH) | 0.568 / 0.613 | 0.637 / 0.690 |
| 4 | A / C v4b | 3-month window instead of year | 0.517 / 0.570 | 0.639 / 0.636 |
| 5 | C v5 | + 11 targeted training regions | 0.626 | 0.722 |
| 5 | C v5b | v4b + 11 regions | 0.580 | 0.598 |

Run names: **A** = U-Net, **B** = DeepLabV3+, **C** = SegFormer; v1 (no number) … v5 = improvement rounds; "b" = alternative idea.

## Folder structure

```
├── preprocess.py           shared data rules (decoding, clipping, 7 classes, colours, normalisation)
├── train_lib.py            training code, round 2  (notebooks 04a-c, 05)
├── train_lib_r3.py         round 3                 (06, 05b)
├── train_lib_r4.py         round 4                 (07, 05c)
├── train_lib_r5.py         rounds 4b / 5           (07b, 08, 05d)
├── notebooks/
│   ├── r1_data_first_model/      01 download, 02 preprocessing, 03 first U-Net
│   ├── r2_model_comparison/      04a U-Net, 04b DeepLabV3+, 04c SegFormer, 05 compare
│   ├── r3_validation_lovasz/     01b, 02b (8 validation regions), 06 training v2/v3, 05b compare
│   ├── r4_year_context/          07 year context (v4), 07b 3-month window (v4b), 05c compare
│   ├── r5_more_regions/          01c, 02c (11 new regions), 08 training v5/v5b, 05d compare
├── data/                   region lists, class statistics  (radar + labels: see "Data")
├── data_r3/ data_r5/       tile index + normalisation statistics per data round (tiles: see "Data")
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
   | [`tiles_r3.zip`](https://zenodo.org/records/24060094/files/tiles_r3.zip?download=1) | 3.2 GB | v2, v3, v4, v4b |
   | [`tiles_r5.zip`](https://zenodo.org/records/24060094/files/tiles_r5.zip?download=1) | 4.1 GB | v5, v5b, v6, v8, U-Net v8 (main dataset) |
   | [`raw_geotiffs.zip`](https://zenodo.org/records/24060094/files/raw_geotiffs.zip?download=1) | 5.4 GB | rebuilding the tiles without Earth Engine |

   Check a download with [`SHA256SUMS.txt`](https://zenodo.org/records/24060094/files/SHA256SUMS.txt?download=1) (PowerShell: `Get-FileHash file.zip`).
2. **Or recreate** it (free Google Earth Engine account; set your project in the download notebooks):
   run the download notebooks, then the preprocessing notebooks, in round order (see `data/README.md`).

Files that **are** in git because the code needs them: region lists (`data/regions*.csv`), tile lists (`data*/patch_index.csv`) and normalisation statistics (`data*/norm_stats*.json`).

## Model weights

`best.pth` files (≈ 95 MB each) are **not in git**. [`weights_all.zip`](https://zenodo.org/records/24060094/files/weights_all.zip?download=1) (1.8 GB) contains
the weights of all runs; unzip it in the project folder (each file lands in its `runs/<run name>/` folder).
Runs up to this round: `A_round2`, `A_unet_resnet34`, `B_deeplabv3plus_resnet50`, `C_segformer_mitb2`, `A_unet_v2`, `B_deeplab_v2`, `C_segformer_v2`, `C_segformer_v3`, `A_unet_v4`, `A_unet_v4b`, `C_segformer_v4`, `C_segformer_v4b`, `C_segformer_v5`, `C_segformer_v5b`.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

All notebooks start with a cell that switches to the project folder, so they can be run from their sub-folder.
