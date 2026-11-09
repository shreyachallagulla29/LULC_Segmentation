"""
Sentinel-1 land-cover prototype (Streamlit).

Input:  latitude / longitude of a place and a start month.
Output: 12 monthly land-cover maps from the final model (SegFormer v8), the % of the area covered by each
        class per month, the month-to-month change and the total change (first -> last month) in
        percentage points, and (optional) ESA WorldCover 2021 for reference.

Run from the project folder:   streamlit run prototype/app.py
"""
import os
import time

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import lulc_core as L
from lulc_core import CLASSES, COLORS, IGNORE, NC

st.set_page_config(page_title="Sentinel-1 Land Cover", page_icon="🛰️", layout="wide")

CMAP = ListedColormap(COLORS + ["#ffffff"])
NORM = BoundaryNorm(list(range(NC + 1)) + [256], NC + 1)
COLOR_SCALE = alt.Scale(domain=CLASSES, range=COLORS)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner=False)
def get_model():
    return L.load_model()


def show_map(ax, m, title):
    ax.imshow(np.where(m == IGNORE, NC, m), cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(title, fontsize=11); ax.axis("off")


def legend(fig, y=0.0):
    fig.legend([Patch(color=c) for c in COLORS] + [Patch(facecolor="white", edgecolor="grey")],
               CLASSES + ["no data"], loc="lower center", ncol=8, frameon=False, fontsize=10, bbox_to_anchor=(0.5, y))


def grid_figure(pred, labels, title=None):
    fig, axs = plt.subplots(3, 4, figsize=(16, 13))
    for k, ax in enumerate(axs.flat):
        show_map(ax, pred[k], labels[k])
    legend(fig)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def style_map(styler, func, **kw):
    """Styler.map (pandas >= 2.1) or Styler.applymap (older pandas)."""
    return styler.map(func, **kw) if hasattr(styler, "map") else styler.applymap(func, **kw)


def fmt_pp(v):
    return "" if pd.isna(v) else f"{v:+.2f}"


def color_pp(v):
    if pd.isna(v) or abs(v) < 0.05:
        return ""
    return "color: #1a7f37" if v > 0 else "color: #d73a49"


# ------------------------------------------------------------------ sidebar: inputs
st.sidebar.header("Location and period")
lat = st.sidebar.number_input("Latitude", -60.0, 75.0, 50.1109, step=0.01, format="%.4f")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 8.6821, step=0.01, format="%.4f")
last_y, last_m = L.latest_start()
c1, c2 = st.sidebar.columns(2)
start_year = c1.selectbox("Start year", list(range(L.FIRST_YEAR, last_y + 1)), index=2021 - L.FIRST_YEAR)
start_month = c2.selectbox("Start month", list(range(1, 13)), format_func=lambda m: L.MONTH_ABBR[m - 1])
size_km = st.sidebar.slider("Area (km × km)", 3, 12, 5,
                            help="Square around the point. Bigger areas take longer (about 25 s for 5 km, "
                                 "2 min for 10 km on a laptop CPU). Pixels are always 10 m.")
months = L.month_list(start_year, start_month)
labels = L.month_labels(months)
too_late = (start_year, start_month) > (last_y, last_m)
st.sidebar.caption(f"12 months: **{labels[0]} – {labels[-1]}**")
if too_late:
    st.sidebar.error(f"The 12 months must be complete. Latest possible start: {L.MONTH_ABBR[last_m - 1]} {last_y}.")

st.sidebar.header("Earth Engine")
project = st.sidebar.text_input("Google Earth Engine project ID", os.environ.get("EE_PROJECT", ""),
                                help="Your Earth Engine Cloud project. Log in once with  earthengine authenticate  in a terminal.")
with_ref = st.sidebar.checkbox("Show ESA WorldCover 2021 for reference", value=False,
                               help="Downloads the ESA WorldCover 2021 map of the same area, for reference only "
                                    "(not used by the model).")
run = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=too_late or not project)

# ------------------------------------------------------------------ header
st.title("🛰️ Land cover from Sentinel-1 radar — 12 months")
st.write("Enter a place and a start month. The app downloads 12 monthly **Sentinel-1 radar** images (VV, VH) "
         "from Google Earth Engine, runs the final model (**SegFormer v8**: all 12 months + year statistics, "
         "test mIoU 0.738, OA 0.846) and maps 7 classes at 10 m.")

top = st.columns([2, 1])
top[0].map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=9, height=220)
top[1].markdown("**Classes**")
top[1].markdown("".join(f"<span style='display:inline-block;width:12px;height:12px;background:{c};"
                        f"margin-right:6px;border:1px solid #999'></span>{n}<br>" for n, c in zip(CLASSES, COLORS)),
                unsafe_allow_html=True)

# ------------------------------------------------------------------ processing
if run:
    status = st.status("Working …", expanded=True)
    bar = st.progress(0.0)
    def progress(f, text):
        bar.progress(min(float(f), 1.0), text=text)
    t0 = time.time()
    try:
        status.write("1/4 Model weights")
        L.ensure_weights(progress)
        model = get_model()
        status.write("2/4 Sentinel-1 download (12 monthly composites, same processing as the training data)")
        s1 = L.download_s1(project, lat, lon, months, size_km, progress)
        status.write("3/4 Running SegFormer v8 on all 12 months")
        pred, conf = L.predict_12_months(model, s1["x12"], progress=progress)
        ref = None
        if with_ref:
            status.write("4/4 ESA WorldCover 2021 (reference)")
            ref = L.download_reference(project, lat, lon, size_km)
        status.update(label=f"Done in {time.time() - t0:.0f} s", state="complete", expanded=False)
        bar.empty()
        st.session_state["res"] = dict(
            lat=lat, lon=lon, size_km=size_km, labels=labels, pred=pred, conf=conf, vv=s1["x12"][:, 0],
            crs=s1["crs"], transform=s1["transform"], orbit=s1["orbit"], images=s1["images"], ref=ref,
            overlap=L.overlap_with_known_regions(lat, lon, size_km), seconds=time.time() - t0)
    except Exception as e:
        status.update(label="Stopped", state="error")
        st.error(str(e))

res = st.session_state.get("res")
if not res:
    st.info("Set the location and start month on the left, enter your Earth Engine project ID and press **Run**.")
    st.stop()

# ------------------------------------------------------------------ results
pred, conf, labels = res["pred"], res["conf"], res["labels"]
area_km2 = res["size_km"] ** 2
shares = L.class_shares(pred, labels)
mom, total = L.changes(shares, area_km2)

st.subheader(f"{res['lat']:.4f}, {res['lon']:.4f} · {res['size_km']} × {res['size_km']} km · {labels[0]} – {labels[-1]}")
info = st.columns(4)
info[0].metric("Area", f"{area_km2} km²")
info[1].metric("Orbit", res["orbit"].title())
info[2].metric("Radar images per month", f"{min(res['images'])}–{max(res['images'])}")
info[3].metric("Processing time", f"{res['seconds']:.0f} s")
if res["overlap"]:
    name, split, frac = res["overlap"][0]
    st.warning(f"This area overlaps the {split} region **{name}** ({frac:.0%} of the area). "
               + ("The model learned from it, so results will look better than on new places." if split == "train"
                  else "It was used to evaluate the model, not to train it."))
if min(res["images"]) == 0:
    st.warning("Some months have no Sentinel-1 image; those months are shown as 'no data'.")

tabs = st.tabs(["🗺️ Maps", "📊 Class percentages", "🔁 Change", "✅ Reference check", "⬇️ Download"])

# --- maps
with tabs[0]:
    st.markdown("**LULC map of the selected location over 12 months**")
    st.pyplot(grid_figure(pred, labels), use_container_width=True)
    st.markdown("**One month in detail**")
    k = st.select_slider("Month", options=list(range(12)), format_func=lambda i: labels[i], value=0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ax[0].imshow(res["vv"][k], cmap="gray", vmin=-25, vmax=5); ax[0].set_title(f"Sentinel-1 VV (dB), {labels[k]}"); ax[0].axis("off")
    show_map(ax[1], pred[k], f"Land cover, {labels[k]}")
    im = ax[2].imshow(conf[k], cmap="viridis", vmin=0.3, vmax=1); ax[2].set_title("Model confidence"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    legend(fig, -0.02); fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.pyplot(fig, use_container_width=True)

# --- class percentages
with tabs[1]:
    st.markdown("**% of the area covered by each class, per month** (of the pixels with radar data in that month)")
    top_share = max(float(np.nanmax(shares.values)), 1.0)
    st.dataframe(style_map(shares.style.format("{:.1f}"),
        lambda v: "" if pd.isna(v) else f"background-color: rgba(26, 127, 55, {0.6 * v / top_share:.2f})"),
        use_container_width=True)
    long = shares.reset_index(names="month").melt("month", var_name="class", value_name="percent")
    chart = (alt.Chart(long).mark_line(point=True)
             .encode(x=alt.X("month:N", sort=labels, title=None), y=alt.Y("percent:Q", title="% of area"),
                     color=alt.Color("class:N", scale=COLOR_SCALE), tooltip=["month", "class", alt.Tooltip("percent:Q", format=".2f")])
             .properties(height=380))
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"1 % of this area = {area_km2 / 100:.2f} km².")

# --- change
with tabs[2]:
    st.markdown("**Total change from the first to the last month** (pp = percentage points: 20 % → 23 % is +3 pp)")
    st.dataframe(style_map(total.style.format({total.columns[0]: "{:.2f}", total.columns[1]: "{:.2f}",
                                               "change (pp)": fmt_pp, "change (km²)": "{:+.3f}"}),
                           color_pp, subset=["change (pp)"]), use_container_width=True)
    bars = (alt.Chart(total.reset_index(names="class"))
            .mark_bar().encode(x=alt.X("class:N", sort=CLASSES, title=None), y=alt.Y("change (pp):Q"),
                               color=alt.Color("class:N", scale=COLOR_SCALE, legend=None),
                               tooltip=["class", alt.Tooltip("change (pp):Q", format="+.2f")])
            .properties(height=300, title=f"Total change {labels[0]} → {labels[-1]} (pp)"))
    st.altair_chart(bars, use_container_width=True)
    st.markdown("**Month-to-month change (pp)**")
    st.dataframe(style_map(mom.style.format(fmt_pp), color_pp), use_container_width=True)
    with st.expander("How reliable is the change? (quality per month)"):
        st.dataframe(L.stability(pred, conf, labels).style.format("{:.2f}"), use_container_width=True)
        st.caption("Real land cover changes little within one year (except crops and seasonal water), so pixels that "
                   "flip class from month to month are mostly model noise. Changes of 1–2 pp or less should not be "
                   "read as real change. The model was trained with one label per year (WorldCover 2021), so it "
                   "tends to keep classes stable over the months.")

# --- reference check (ESA WorldCover 2021, for reference only)
with tabs[3]:
    ref = res["ref"]
    if ref is None:
        st.info("Tick **Show ESA WorldCover 2021 for reference** on the left and press Run again.")
    else:
        st.markdown("**ESA WorldCover 2021 — for reference only.**")
        rows = []
        for k in range(12):
            a = L.agreement(pred[k], ref["worldcover"])
            rows.append({"month": labels[k], "OA": a["OA"], "mIoU": a["mIoU"]})
        agree = pd.DataFrame(rows).set_index("month")
        st.markdown("**Agreement with ESA WorldCover 2021 per month** (mIoU over classes covering ≥ 1 % of WorldCover)")
        st.dataframe(agree.style.format("{:.3f}"), use_container_width=True)
        k = st.select_slider("Month to compare", options=list(range(12)), format_func=lambda i: labels[i], value=0, key="refk")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6.4))
        show_map(ax[0], pred[k], f"SegFormer v8 (radar), {labels[k]}")
        show_map(ax[1], ref["worldcover"], "ESA WorldCover 2021")
        legend(fig, -0.02); fig.tight_layout(rect=(0, 0.05, 1, 1))
        st.pyplot(fig, use_container_width=True)
        comp = pd.DataFrame({"SegFormer v8 (mean of 12 months) %": shares.mean(),
                             "ESA WorldCover 2021 %": L.class_shares(ref["worldcover"][None], ["wc"]).iloc[0]})
        st.markdown("**Class percentages compared**")
        st.dataframe(comp.style.format("{:.1f}"), use_container_width=True)

# --- download
with tabs[4]:
    tag = f"lulc_{res['lat']:.4f}_{res['lon']:.4f}_{labels[0].replace(' ', '')}"
    st.download_button("Class percentages per month (CSV)", shares.round(3).to_csv().encode(), f"{tag}_percentages.csv", "text/csv")
    st.download_button("Month-to-month change in pp (CSV)", mom.round(3).to_csv().encode(), f"{tag}_month_to_month_pp.csv", "text/csv")
    st.download_button("Total change, first → last month (CSV)", total.round(3).to_csv().encode(), f"{tag}_total_change.csv", "text/csv")
    st.download_button("12 monthly maps (GeoTIFF, 1 band per month)",
                       L.to_geotiff(pred, res["crs"], res["transform"], labels), f"{tag}_maps.tif", "image/tiff")
    st.caption("GeoTIFF values: " + ", ".join(f"{i} = {c}" for i, c in enumerate(CLASSES)) + ", 255 = no data. "
               "It opens in QGIS with the class colours.")
