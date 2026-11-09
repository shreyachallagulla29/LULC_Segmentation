# Prototype — 12-month land-cover maps from Sentinel-1

A Streamlit app that runs the final model (**SegFormer v8**) on any place on Earth.

| | |
|---|---|
| **Input** | latitude / longitude of a location, a start month (any month from Jan 2015 on), area size (3–12 km) |
| **Processing** | downloads 12 monthly Sentinel-1 composites from Google Earth Engine (same processing as the training data) and runs the trained model on all 12 months |
| **Calculation** | % of the area covered by each class for every month; month-to-month change and total change (first → last month) in **percentage points** (pp) |
| **Visualisation** | the land-cover map of the location for each of the 12 months, one month in detail (radar, map, confidence), line and bar charts |
| **Extras** | optional ESA WorldCover 2021 map of the same area, for reference only; CSV and GeoTIFF download |

## Setup (once)

From the **project folder** (one level above this one):

```bash
pip install -r prototype/requirements.txt
earthengine authenticate          # log in to Google Earth Engine (opens the browser)
```

You need a Google Earth Engine Cloud project (free for research / education): <https://earthengine.google.com>.

**Model weights:** the app uses `runs/C_segformer_v8_border/best.pth`. If that file is not there, the app downloads
[`weights_final.zip`](https://zenodo.org/records/24060094/files/weights_final.zip?download=1) (0.5 GB) from Zenodo
([doi.org/10.5281/zenodo.24060094](https://doi.org/10.5281/zenodo.24060094)) on the first run and unpacks only the
final model — the original full-precision weights, unchanged.

## Run

```bash
streamlit run prototype/app.py
```

The app opens in the browser. On the left: enter latitude, longitude, start year and month, area size and your
Earth Engine project ID (or set it once with the environment variable `EE_PROJECT`), then press **Run**.

Time on a laptop CPU: about 25 s of model time for 5 × 5 km, about 2 min for 10 × 10 km, plus the
Earth Engine download (usually a minute or two). Downloads are cached in `prototype/cache/`, so running the same place again is fast.

## How the numbers are calculated

* **Class share (%)** of month *m* = pixels of the class ÷ pixels with radar data in month *m* × 100.
* **Month-to-month change (pp)** = share in month *m* − share in month *m − 1*.
* **Total change (pp)** = share in the last month − share in the first month (also given in km²).
  Example: Cropland 20 % → 23 % = **+3 pp**.
* The **quality** table (Change tab) shows radar coverage, mean model confidence and how many pixels changed class since
  the previous month. Real land cover changes little within one year, so changes of about 1–2 pp or less are mostly noise.

## Files

| File | What it does |
|---|---|
| `app.py` | the user interface (Streamlit) |
| `lulc_core.py` | download (Earth Engine), model loading, prediction over the area, percentages and change |

The model and preprocessing code are imported from the project (`preprocess.py`, `train_lib_r8.py`), so the app
prepares the data exactly like the training notebooks.

## Limitations

* Trained on 2021 data; other years work, but from 2022 there are fewer radar images per month (Sentinel-1B stopped in Dec 2021).
* Grass/Shrub and Wetland are the weakest classes; flooded rice paddies can be mapped as Wetland.
* The model was trained with one label per year, so maps are stable over the months and cautious about real change.
* Snow and ice are not one of the 7 classes.
