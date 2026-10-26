# Data

Inputs: **Sentinel-1 GRD** (IW, VV + VH, 10 m) monthly composites for 2021, from Google Earth Engine
(`COPERNICUS/S1_GRD`). Labels: **ESA WorldCover 2021** (v200), remapped to 7 classes.
Each region is 12 x 12 km (1200 x 1200 px), cut into 256 x 256 px tiles.

Not in git (too large) — archived on Zenodo: **[doi.org/10.5281/zenodo.24060094](https://doi.org/10.5281/zenodo.24060094)**.
In git: region lists, tile lists and statistics (needed by the code).

Unzip every package **in the project folder**. Raw GeoTIFFs of all regions (instead of downloading from Earth Engine): [`raw_geotiffs.zip`](https://zenodo.org/records/24060094/files/raw_geotiffs.zip?download=1).

## What each round uses

| Round / models | Region list | Tile folder | Recreate with (in this order) | Download package |
|---|---|---|---|---|
| 1-2 (first U-Net, v1) | `data/regions.csv` | `data/patches/` | `r1_data_first_model/01_download_data`, `02_preprocessing` | [`tiles_r1-2.zip`](https://zenodo.org/records/24060094/files/tiles_r1-2.zip?download=1) |
| 3-4b (v2, v3, v4, v4b) | `data/regions_r3.csv` | `data_r3/patches/` | + `r3_validation_lovasz/01b_download_r3`, `02b_preprocessing_r3` | [`tiles_r3.zip`](https://zenodo.org/records/24060094/files/tiles_r3.zip?download=1) |
| 5, 6, 8 (v5, v5b, v6, v8, U-Net v8) | `data/regions_r5.csv` | `data_r5/patches/` | + `r5_more_regions/01c_download_r5`, `02c_preprocessing_r5` | [`tiles_r5.zip`](https://zenodo.org/records/24060094/files/tiles_r5.zip?download=1) |

Downloads go to `data/raw/` (radar `<region>_s1_2021.tif` + label `<region>_label.tif`).

## Licences

Contains modified Copernicus Sentinel data (2021). ESA WorldCover 2021, © ESA WorldCover project, CC BY 4.0.
