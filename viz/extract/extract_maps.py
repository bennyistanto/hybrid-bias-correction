"""Render the gridded QA climatologies as compact PNG map tiles for the viz.

For each correction stage (LS, LSEQM, LSEQM+DL) and each quality dimension, take
the annual-mean over the 36 dekads of the gauge-referenced QA grid
(`qualitysd_cpc`, 0.1 deg, 171 x 461) and render a coastline-overlaid PNG. Colour
scales are fixed per dimension across stages, so toggling the stage is a true
like-for-like "morph". Also renders a DL-minus-LS difference tile per dimension.

    python viz/extract/extract_maps.py

Writes: viz/src/data/maps/<dim>_<stage>.png, <dim>_diff.png, and map_meta.json.
"""
import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd

ROOT = Path(r"C:\Users\benny\OneDrive\Documents\Github\hybrid-bias-correction")
OUT = ROOT / "viz" / "src" / "data"
MAPS = OUT / "maps"
MAPS.mkdir(parents=True, exist_ok=True)
BBOX = (93.0, -13.0, 143.0, 8.0)

STAGES = ["ls", "lseqm", "lseqmdl"]
STAGE_LABEL = {"ls": "LS", "lseqm": "LSEQM", "lseqmdl": "LSEQM+DL"}
# dim key -> (netcdf var, short label, "seq"|"cat")
DIMS = {
    "cqi":          ("continuous_quality",         "Continuous Quality Index", "seq"),
    "confidence":   ("confidence_level",           "Confidence level",         "seq"),
    "categorical":  ("categorical_quality",        "Categorical class",        "cat"),
    "basic":        ("basic_statistical_quality",  "Basic statistical",        "seq"),
    "distribution": ("distribution_quality",       "Distribution",             "seq"),
    "temporal":     ("temporal_quality",           "Temporal",                 "seq"),
}
CAT_COLORS = ["#d73027", "#fee08b", "#1a9850"]  # class 1 (poor) -> 3 (good)
SEQ_CMAP = "RdYlGn"   # low quality red -> high quality green
DIFF_CMAP = "BrBG"    # brown = DL lowered, green = DL raised

# coastlines (dissolve to a single boundary layer over the map extent)
adm0 = gpd.read_file(ROOT / "data/subset/bnd/wld_bnd_adm0.shp", bbox=BBOX)


def clim(stage, var):
    """Annual-mean climatology of one QA var for one stage (gauge-referenced)."""
    files = sorted(glob.glob(str(
        ROOT / f"data/output/quality_{stage}/idn_cli_qualitysd_cpc_imergl_{stage}_*.nc4")))
    if not files:
        raise FileNotFoundError(f"no qualitysd_cpc files for stage {stage}")
    ds0 = xr.open_dataset(files[0])
    lat, lon = ds0.lat.values, ds0.lon.values
    stack = np.stack([xr.open_dataset(f)[var].values for f in files])
    return np.nanmean(stack, axis=0), lat, lon


def nan_smooth(a, sigma=1.0):
    """NaN-aware Gaussian smoothing for display: denoises the per-pixel QA
    speckle while preserving the exact land mask (no bleed across coastlines)."""
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), sigma, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 1e-6)
    out[~m] = np.nan
    return out


def render(grid, extent, vmin, vmax, cmap, path, interp="bilinear", points=None):
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig = plt.figure(figsize=(9, 9 / aspect), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    cm = (plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap)
    cm.set_bad(alpha=0.0)  # NaN (ocean / inland water) stays transparent
    ax.imshow(np.ma.masked_invalid(grid), extent=extent, origin="lower",
              vmin=vmin, vmax=vmax, cmap=cm, interpolation=interp)
    adm0.boundary.plot(ax=ax, color="#3a3a3a", linewidth=0.4)
    if points is not None:  # station locations overlaid as small dots
        ax.scatter(points[:, 0], points[:, 1], s=7, facecolors="white",
                   edgecolors="#222222", linewidths=0.4, alpha=0.9, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.savefig(path, transparent=True, dpi=170)
    plt.close(fig)


meta = {"dims": {}, "means": {}, "stage_label": STAGE_LABEL, "diff_pair": "LSEQM+DL − LS"}
for dk, (var, label, kind) in DIMS.items():
    grids, lat, lon = {}, None, None
    for stg in STAGES:
        g, lat, lon = clim(stg, var)
        grids[stg] = g
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # Domain statistics are computed on the RAW grids; only the display is smoothed.
    # Pooling must match paper/NUMBERS.md S24: the spatial MEDIAN over finite pixels
    # within each dekad, then the mean of those 36 medians. A spatial nanmean was used
    # here previously and produced a third published CQI triple (0.536 / 0.500 / 0.503)
    # alongside the ledger's and the docs site's, for what a reader takes to be one
    # quantity. Emitted under an explicit key so the pooling travels with the number.
    meta["means"][dk] = {stg: round(float(np.nanmedian(g)), 3) for stg, g in grids.items()}
    meta.setdefault("pooling", {})[dk] = "spatial median over finite pixels, dekad-averaged"

    cat = kind == "cat"
    interp = "nearest" if cat else "bilinear"
    if cat:
        cmap, vmin, vmax = ListedColormap(CAT_COLORS), 0.5, 3.5
    else:
        allv = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
        vmin, vmax = float(np.percentile(allv, 2)), float(np.percentile(allv, 98))
        cmap = SEQ_CMAP

    # The QA carries a 0.5-deg lattice (the CPC reference is 0.5 deg, upsampled
    # to 0.1 deg), strongest in the corrected stages. A 0.5-deg period needs
    # sigma ~ 2 px (0.2 deg) to erase, so smooth every sequential field there.
    disp = {stg: (g if cat else nan_smooth(g, 2.0)) for stg, g in grids.items()}
    for stg, g in disp.items():
        render(g, extent, vmin, vmax, cmap, MAPS / f"{dk}_{stg}.png", interp)

    diff = grids["lseqmdl"] - grids["ls"]
    dabs = float(np.percentile(np.abs(diff[np.isfinite(diff)]), 98)) or 0.01
    render(nan_smooth(diff, 2.0), extent, -dabs, dabs, DIFF_CMAP, MAPS / f"{dk}_diff.png", "bilinear")

    meta["dims"][dk] = {
        "var": var, "label": label, "kind": kind,
        "vmin": round(vmin, 3), "vmax": round(vmax, 3), "diff_abs": round(dabs, 3),
    }
    print(f"  {dk}: vmin={vmin:.3f} vmax={vmax:.3f} means={meta['means'][dk]}")

meta["extent"] = extent
(OUT / "map_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# ---------------------- station-density confidence mask (5.4) --------------
# The mask that gates the CNN blend. Render the confidence field (land-masked,
# with the 180 gauges overlaid), and export a counts histogram so the page can
# recompute the blend for other saturation counts.
BASE_ALPHA = 0.70
SATURATION = 2.0  # operating default (config.yml, config.py, and the nc4 attr)
dens = xr.open_dataset(
    ROOT / "data/output/station_density/confidence_mask_station_density.nc4")
conf = dens["confidence"].values
latd, lond = dens.lat.values, dens.lon.values
dext = [float(lond.min()), float(lond.max()), float(latd.min()), float(latd.max())]

land = np.isfinite(clim("lseqmdl", "continuous_quality")[0])  # QA valid = land
conf_land = np.where(land, conf, np.nan)

stjson = json.loads((OUT / "stations.json").read_text(encoding="utf-8"))
pts = np.array([[s["lon"], s["lat"]] for s in stjson])

render(nan_smooth(conf_land, 2.0), dext, 0.0, float(np.nanmax(conf_land)), "viridis",
       MAPS / "density_confidence.png", interp="bilinear", points=pts)

lv = conf_land[land]                      # land confidence values
counts = lv * SATURATION                  # back out smoothed station counts
dl = lv * (1.0 - BASE_ALPHA)              # DL weight = confidence x 0.30
hist, edges = np.histogram(counts, bins=48, range=(0.0, float(counts.max())))
density = {
    "base_alpha": BASE_ALPHA, "saturation_count": SATURATION,
    "sigma_deg": 0.5, "sigma_km": 55, "n_stations": 180, "n_land": int(land.sum()),
    "max_confidence": round(float(np.nanmax(lv)), 3),
    "mean_confidence": round(float(np.nanmean(lv)), 3),
    "pct_dl_active": round(float((lv > 0.001).mean() * 100), 1),
    "max_dl_weight_pct": round(float(np.nanmax(dl) * 100), 1),
    "mean_dl_weight_pct": round(float(np.nanmean(dl) * 100), 2),
    # histogram of smoothed station counts over land, for the saturation what-if
    "hist_counts": hist.tolist(),
    "hist_edges": [round(float(e), 3) for e in edges],
}
(OUT / "density.json").write_text(json.dumps(density, indent=2), encoding="utf-8")
print(f"  density: max_conf={density['max_confidence']} "
      f"mean_dl={density['mean_dl_weight_pct']}% active={density['pct_dl_active']}%")

print("Done. PNG tiles + map_meta.json + density in", MAPS)
