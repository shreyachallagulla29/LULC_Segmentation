# Sentinel-1 Land-Cover Mapping (radar only)

Semantic segmentation of **7 land-cover classes** (Trees, Grass/Shrub, Cropland, Built-up, Bare, Water, Wetland)
from **Sentinel-1 radar (VV, VH)** at 10 m, with **ESA WorldCover 2021** as labels.
The final model maps any place for 12 months and reports class percentages and their change.

**Final model: SegFormer v8** (weights: `runs/C_segformer_v8_border/best.pth`, [download](https://zenodo.org/records/24060094/files/weights_final.zip?download=1)) — validation mIoU 0.654, test mIoU 0.738 (OA 0.846).

## Folder structure

```
├── preprocess.py           shared data rules (decoding, clipping, 7 classes, colours, normalisation)
├── train_lib.py            training code, round 2  (notebooks 04a-c, 05)
├── train_lib_r3.py         round 3                 (06, 05b)
├── train_lib_r4.py         round 4                 (07, 05c)
├── train_lib_r5.py         rounds 4b / 5           (07b, 08, 05d)
├── train_lib_r6.py         rounds 6 / 7            (16, 18) + predict_region() for whole areas
├── train_lib_r8.py         round 8                 (19, 19b): label-border ignore, U-Net / DeepLabV3+ option
├── notebooks/
│   ├── r1_data_first_model/      01 download, 02 preprocessing, 03 first U-Net
│   ├── r2_model_comparison/      04a U-Net, 04b DeepLabV3+, 04c SegFormer, 05 compare
│   ├── r3_validation_lovasz/     01b, 02b (8 validation regions), 06 training v2/v3, 05b compare
│   ├── r4_year_context/          07 year context (v4), 07b 3-month window (v4b), 05c compare
│   ├── r5_more_regions/          01c, 02c (10 new regions), 08 training v5/v5b, 05d compare
│   ├── r6_temporal/              16 all 12 months + 18 year statistics (v6)
│   ├── r7_rice_wetland_regions/  17a, 17b (20 new regions), 18 training v7
│   ├── r8_label_borders/         19 SegFormer v8, 19b U-Net v8
│   └── final_model.ipynb         the final model: curves, val / test results, confusion, comparison, example maps
├── prototype/              Streamlit app: place + start month -> 12 monthly maps, % per class, change (pp)
├── data/                   region lists, class statistics  (radar + labels: see "Data")
├── data_r3/ data_r5/ data_r7/   tile index + normalisation statistics per data round (tiles: see "Data")
└── runs/                   one folder per trained model: config, metrics, curves, confusion matrices
                            (weights *.pth: see "Model weights")
```

Run names: **A** = U-Net, **B** = DeepLabV3+, **C** = SegFormer; v1 (no number) … v8 = improvement rounds; "b" = alternative idea.
All notebooks start with a cell that switches to the project folder, so run them from anywhere inside it.

## Results (same 8 validation and 6 test regions for all models)

| Round | Run | What changed | Val mIoU | Test mIoU |
|---|---|---|---|---|
| 2 | A / B / C (v1) | U-Net vs DeepLabV3+ vs SegFormer | 0.430 / 0.497 / 0.502 | 0.577 / 0.605 / 0.534 |
| 3 | A / B / C v2 | 8 val regions, Lovász loss, longer training | 0.488 / 0.500 / 0.536 | 0.625 / 0.610 / 0.616 |
| 3 | C v3 | v2 + regularisation | 0.543 | 0.618 |
| 4 | A / C v4 | + year context (12-month mean / std of VV, VH) | 0.568 / 0.613 | 0.637 / 0.690 |
| 4 | A / C v4b | 3-month window instead of year | 0.517 / 0.570 | 0.639 / 0.636 |
| 5 | C v5 | + 10 targeted training regions | 0.626 | 0.722 |
| 5 | C v5b | v4b + 10 regions | 0.580 | 0.598 |
| 6 | C v6 | all 12 months + 18 year statistics (temporal encoder) | 0.650 | 0.730 |
| 7 | C v7 | v6 + 20 regions (rice, wetland, shrub, cities) | 0.642 | 0.676 |
| 8 | **C v8** | **v6 + label borders ignored in training** | **0.654** | **0.738** |
| 8 | A v8 | same as v8 with U-Net | 0.616 | 0.668 |

Known limitation: flooded rice paddies are often classified as Wetland (v7 fixes this but loses natural wetland).

## Data

The data is **not in git** (≈ 14 GB). It is archived on Zenodo: **[doi.org/10.5281/zenodo.24060094](https://doi.org/10.5281/zenodo.24060094)**.
Two ways to get it — see [`data/README.md`](data/README.md) for details per round:

1. **Download** the prepared packages and unzip each one **in the project folder** (the paths inside the zips put every file in place):

   | Package | Size | Needed for |
   |---|---|---|
   | [`tiles_r5.zip`](https://zenodo.org/records/24060094/files/tiles_r5.zip?download=1) | 4.1 GB | v5, v5b, v6, v8, U-Net v8 (main dataset) |
   | [`tiles_r7_added.zip`](https://zenodo.org/records/24060094/files/tiles_r7_added.zip?download=1) | 1.7 GB | v7 (together with `tiles_r5`) |
   | [`tiles_r3.zip`](https://zenodo.org/records/24060094/files/tiles_r3.zip?download=1) | 3.2 GB | v2, v3, v4, v4b |
   | [`tiles_r1-2.zip`](https://zenodo.org/records/24060094/files/tiles_r1-2.zip?download=1) | 3.0 GB | first U-Net, v1 |
   | [`raw_geotiffs.zip`](https://zenodo.org/records/24060094/files/raw_geotiffs.zip?download=1) | 5.4 GB | rebuilding all tiles without Earth Engine (notebooks 02*, 17b) |

   Check a download with [`SHA256SUMS.txt`](https://zenodo.org/records/24060094/files/SHA256SUMS.txt?download=1) (PowerShell: `Get-FileHash file.zip`).
2. **Recreate** with the notebooks (needs a free Google Earth Engine account): download notebooks
   (`01*`, `17a`) then preprocessing notebooks (`02*`, `17b`), in round order.

Files that **are** in git because the code needs them: region lists (`data/regions*.csv`), tile lists
(`data_r*/patch_index.csv`) and normalisation statistics (`data_r*/norm_stats*.json`).

## Model weights

### Final model (SegFormer v8)

| | |
|---|---|
| **File name** | `best.pth` (99 MB, full precision) |
| **Where it goes** | `runs/C_segformer_v8_border/best.pth` |
| **Download** | [`weights_final.zip`](https://zenodo.org/records/24060094/files/weights_final.zip?download=1) on Zenodo ([doi.org/10.5281/zenodo.24060094](https://doi.org/10.5281/zenodo.24060094)) |

Unzip `weights_final.zip` in the project folder; the final model lands at `runs/C_segformer_v8_border/best.pth`.
The prototype downloads it automatically on its first run if the file is missing.

### All weights

`best.pth` files (≈ 95 MB each) are **not in git**. Download from Zenodo and unzip in the project folder (each file lands in its `runs/<run name>/` folder):

- [`weights_final.zip`](https://zenodo.org/records/24060094/files/weights_final.zip?download=1) (0.5 GB): v8 (final), v6, v7, v5, U-Net v8 — enough to run the prototype and `notebooks/final_model.ipynb`
- [`weights_all.zip`](https://zenodo.org/records/24060094/files/weights_all.zip?download=1) (1.8 GB): all 18 runs

## Prototype

`prototype/app.py` (Streamlit) runs the final model on any place: enter **latitude / longitude and a start month** and it
downloads 12 monthly Sentinel-1 composites from Google Earth Engine, maps all 12 months, and reports the **% of the area per
class for every month**, the **month-to-month change** and the **total change** (first → last month) in percentage points.
Optional: ESA WorldCover 2021 map of the same area, for reference only. Details and setup: [`prototype/README.md`](prototype/README.md).

```bash
pip install -r prototype/requirements.txt
earthengine authenticate
streamlit run prototype/app.py
```

The final model weights (`runs/C_segformer_v8_border/best.pth`, full precision) come from Zenodo (`weights_final.zip`);
the app downloads them automatically on the first run if they are not in the project folder.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Final model summary with all graphs: `notebooks/final_model.ipynb`.
