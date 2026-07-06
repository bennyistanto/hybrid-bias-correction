"""Extract compact static data for the Observable Framework viz.

Reads the thesis inputs/outputs and writes small JSON / GeoJSON files into
`viz/src/data/`, which the Observable pages load with FileAttachment. Run from
anywhere with the `climate` conda env:

    python viz/extract/extract_core.py

Headline metric values are transcribed from the verified thesis tables
(Table 4.1 vs BMKG, Table 4.2 temporal rows) - the authoritative pooled numbers
- so the dashboard and the manuscript never disagree. Everything else is read
straight from the source data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

ROOT = Path(r"C:\Users\benny\OneDrive\Documents\Github\hybrid-bias-correction")
OUT = ROOT / "viz" / "src" / "data"
OUT.mkdir(parents=True, exist_ok=True)

BBOX = (93.0, -13.0, 143.0, 8.0)  # lon0, lat0, lon1, lat1 - matches Figure 3.1


def write_json(name, obj, indent=None):
    # allow_nan=False forces valid JSON; we pre-sanitise NaN -> None everywhere.
    (OUT / name).write_text(json.dumps(obj, indent=indent, allow_nan=False), encoding="utf-8")
    print(f"  wrote {name}")


def jnum(v, nd):
    """Round to nd decimals, mapping NaN/inf to None (valid JSON null)."""
    v = float(v)
    return None if not np.isfinite(v) else round(v, nd)


# ---------------------------------------------------------------- 1. stations
st = pd.read_csv(ROOT / "data/input/stations/idn_cli_weatherstation_location_bmkg.csv", sep=";")
st.columns = [c.strip().lstrip("﻿") for c in st.columns]
tzname = {7: "WIB", 8: "WITA", 9: "WIT"}
stations = [
    {"id": int(r.ID), "wmo": int(r.ID_WMO), "name": str(r.Station),
     "lon": round(float(r.Lon), 4), "lat": round(float(r.Lat), 4),
     "elev": float(r.Elevation), "region": str(r.region), "prov": str(r.a1name),
     "tz": tzname.get(int(r.timezone), str(r.timezone))}
    for r in st.itertuples()
]
write_json("stations.json", stations)

# ------------------------------------------------------------- 2. boundaries
adm1 = gpd.read_file(ROOT / "data/subset/bnd/idn_bnd_adm1.shp", bbox=BBOX)
adm1 = gpd.clip(adm1, box(*BBOX))  # trim geometry to the bbox, not just filter features
adm1["geometry"] = adm1.geometry.simplify(0.05)  # lighter paths -> snappy web map
adm1 = adm1[["a1name", "geometry"]].rename(columns={"a1name": "province"})
(OUT / "idn_adm1.geojson").unlink(missing_ok=True)
adm1.to_file(OUT / "idn_adm1.geojson", driver="GeoJSON", COORDINATE_PRECISION=3)
print("  wrote idn_adm1.geojson")

wld = gpd.read_file(ROOT / "data/subset/bnd/wld_bnd_adm0.shp", bbox=BBOX)
nc = next((c for c in wld.columns
           if wld[c].astype(str).str.contains("Indonesia", case=False, na=False).any()), None)
neigh = wld[~wld[nc].astype(str).str.contains("Indonesia", case=False, na=False)] if nc else wld
neigh = gpd.clip(neigh, box(*BBOX)).copy()  # trim to bbox so the map extent is Indonesia
neigh["geometry"] = neigh.geometry.simplify(0.03)
(OUT / "neighbours.geojson").unlink(missing_ok=True)
neigh[["geometry"]].to_file(OUT / "neighbours.geojson", driver="GeoJSON", COORDINATE_PRECISION=3)
print("  wrote neighbours.geojson")

# --------------------------------------------- 3. window-offset r(h) curves
z = np.load(ROOT / "temp/subdaily_lag/convention_conflict.npz")
window = {
    "h": [int(x) for x in z["H"]],
    "cpc": [round(float(x), 4) for x in z["r_cpc"]],
    "bmkg": [round(float(x), 4) for x in z["r_bmkg"]],
    "cpc_relabelled": [round(float(x), 4) for x in z["r_cpc_relabelled"]],
}
write_json("window_curves.json", window)

# -------------------------------------- 4. headline metrics (verified thesis)
headline = {
    "stats": {
        "stations": 172, "archived": 180, "pixels": 19395, "cpc_cells": 1256,
        "dekads": 36, "period_cpc": "2001-2025", "period_bmkg": "2001-2021",
        "repro_min": 72.1, "r_flat": 0.34,
        "r_window_utc": 0.20, "r_window_local": 0.57, "offset_h": -23,
    },
    # Table 4.1 - out-of-sample skill vs 172 BMKG stations
    "bmkg": [
        {"pillar": "Value Adjustment",       "metric": "Relative Bias",       "ls": -0.114, "lseqm": 0.009, "lseqmdl": -0.006, "target": 0.0,  "goal": "zero"},
        {"pillar": "Value Adjustment",       "metric": "SDR",                 "ls": 0.71,  "lseqm": 1.03,  "lseqmdl": 1.00,  "target": 1.0,  "goal": "one"},
        {"pillar": "Distribution Alignment", "metric": "KS p-value (%)",      "ls": 0.01,  "lseqm": 19.07, "lseqmdl": 19.07, "target": 100.0,"goal": "high"},
        {"pillar": "Distribution Alignment", "metric": "Wet-Day Freq. Ratio", "ls": 1.21,  "lseqm": 0.96,  "lseqmdl": 0.95,  "target": 1.0,  "goal": "one"},
        {"pillar": "Distribution Alignment", "metric": "Wet-Day Int. Ratio",  "ls": 0.71,  "lseqm": 1.06,  "lseqmdl": 1.05,  "target": 1.0,  "goal": "one"},
        {"pillar": "Extreme Preservation",   "metric": "Q95 Ratio",           "ls": 0.74,  "lseqm": 1.07,  "lseqmdl": 1.05,  "target": 1.0,  "goal": "one"},
        {"pillar": "Extreme Preservation",   "metric": "Q99 Ratio",           "ls": 0.71,  "lseqm": 1.05,  "lseqmdl": 1.01,  "target": 1.0,  "goal": "one"},
        {"pillar": "Event Detection",        "metric": "POD",                 "ls": 0.78,  "lseqm": 0.65,  "lseqmdl": 0.65,  "target": 1.0,  "goal": "high"},
        {"pillar": "Event Detection",        "metric": "CSI",                 "ls": 0.53,  "lseqm": 0.49,  "lseqmdl": 0.49,  "target": 1.0,  "goal": "high"},
        {"pillar": "Event Detection",        "metric": "FAR",                 "ls": 0.36,  "lseqm": 0.32,  "lseqmdl": 0.32,  "target": 0.0,  "goal": "zero"},
    ],
    # Table 4.2 - in-sample skill vs CPC-UNI (the calibration target), full
    "cpc": [
        {"pillar": "Value Adjustment",       "metric": "Relative Bias",       "ls": 0.000,  "lseqm": 0.070,  "lseqmdl": 0.065,  "target": 0.0,   "goal": "zero"},
        {"pillar": "Value Adjustment",       "metric": "SDR",                 "ls": 0.971,  "lseqm": 1.159,  "lseqmdl": 1.148,  "target": 1.0,   "goal": "one"},
        {"pillar": "Distribution Alignment", "metric": "KS p-value (%)",      "ls": 6.56,   "lseqm": 0.00,   "lseqmdl": 0.00,   "target": 100.0, "goal": "high"},
        {"pillar": "Distribution Alignment", "metric": "Wet-Day Freq. Ratio", "ls": 1.040,  "lseqm": 0.947,  "lseqmdl": 0.946,  "target": 1.0,   "goal": "one"},
        {"pillar": "Distribution Alignment", "metric": "Wet-Day Int. Ratio",  "ls": 0.961,  "lseqm": 1.157,  "lseqmdl": 1.152,  "target": 1.0,   "goal": "one"},
        {"pillar": "Extreme Preservation",   "metric": "Q95 Ratio",           "ls": 0.946,  "lseqm": 1.171,  "lseqmdl": 1.163,  "target": 1.0,   "goal": "one"},
        {"pillar": "Extreme Preservation",   "metric": "Q99 Ratio",           "ls": 0.967,  "lseqm": 1.212,  "lseqmdl": 1.196,  "target": 1.0,   "goal": "one"},
        {"pillar": "Event Detection",        "metric": "POD",                 "ls": 0.767,  "lseqm": 0.707,  "lseqmdl": 0.707,  "target": 1.0,   "goal": "high"},
        {"pillar": "Event Detection",        "metric": "CSI",                 "ls": 0.589,  "lseqm": 0.572,  "lseqmdl": 0.572,  "target": 1.0,   "goal": "high"},
        {"pillar": "Event Detection",        "metric": "FAR",                 "ls": 0.271,  "lseqm": 0.248,  "lseqmdl": 0.248,  "target": 0.0,   "goal": "zero"},
        {"pillar": "Temporal Skill",         "metric": "Pearson r",           "ls": 0.343,  "lseqm": 0.345,  "lseqmdl": 0.348,  "target": 1.0,   "goal": "high"},
        {"pillar": "Temporal Skill",         "metric": "RMSE (mm/day)",       "ls": 13.10,  "lseqm": 14.18,  "lseqmdl": 14.07,  "target": 0.0,   "goal": "zero"},
        {"pillar": "Temporal Skill",         "metric": "NSE",                 "ls": -0.273, "lseqm": -0.548, "lseqmdl": -0.524, "target": 1.0,   "goal": "high"},
    ],
    # Table 4.2 temporal rows kept separately (used by the r-ceiling page)
    "temporal": [
        {"metric": "Pearson r",     "ls": 0.343,  "lseqm": 0.345,  "lseqmdl": 0.348},
        {"metric": "RMSE (mm/day)", "ls": 13.10,  "lseqm": 14.18,  "lseqmdl": 14.07},
        {"metric": "NSE",           "ls": -0.273, "lseqm": -0.548, "lseqmdl": -0.524},
    ],
}
write_json("headline.json", headline, indent=2)

# ------------------------------ 5. per-station pooled metrics (Taylor + card)
# One row per station per stage: the mean over the 36 dekads of each metric,
# joined to region / name / coords from the station table (by WMO id).
meta = {int(r.ID_WMO): {"name": str(r.Station), "region": str(r.region),
                        "lon": round(float(r.Lon), 3), "lat": round(float(r.Lat), 3)}
        for r in st.itertuples()}
KEEP = ["pearson_correlation", "stdev_ratio", "relative_bias", "rmse",
        "pod", "csi", "far", "nse", "ks_pvalue"]
SV = ROOT / "data/output/station_validation"
station_metrics = {}
for skey in ["ls", "lseqm", "lseqmdl"]:
    files = sorted(SV.glob(f"station_validation_{skey}_*.csv"))
    alldf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    grp = alldf.groupby("station_id")[KEEP].mean()
    recs = []
    for wmo, row in grp.iterrows():
        m = meta.get(int(wmo), {})
        recs.append({
            "id": int(wmo), "name": m.get("name", str(wmo)), "region": m.get("region", "Unknown"),
            "lon": m.get("lon"), "lat": m.get("lat"),
            "r": jnum(row.pearson_correlation, 3), "sdr": jnum(row.stdev_ratio, 3),
            "bias": jnum(row.relative_bias, 3), "rmse": jnum(row.rmse, 2),
            "pod": jnum(row.pod, 3), "csi": jnum(row.csi, 3),
            "far": jnum(row.far, 3), "nse": jnum(row.nse, 3),
            "ks": jnum(row.ks_pvalue, 3),
        })
    station_metrics[skey] = recs
    print(f"  station_metrics[{skey}]: {len(recs)} stations")
write_json("station_metrics.json", station_metrics)

# ------------------------- 6. threshold-detection curves (Skill page)
# Detection skill vs rainfall threshold, pooled: mean over the 36 dekads of the
# per-dekad cross-station median (and p25/p75 band) at each threshold, per stage.
THR_METRICS = ["pod", "csi", "far", "ets", "hss", "fbi"]
threshold_curves = {}
for skey in ["ls", "lseqm", "lseqmdl"]:
    files = sorted(SV.glob(f"multi_threshold_summary_{skey}_*.csv"))
    alldf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    grp = alldf.groupby("threshold_mm", as_index=False).mean(numeric_only=True)
    labelmap = alldf.drop_duplicates("threshold_mm").set_index("threshold_mm")["label"].to_dict()
    rows = []
    for _, r in grp.iterrows():
        thr = r.threshold_mm
        rec = {"thr": int(thr), "label": str(labelmap[thr]).replace("_", " "),
               "events": jnum(r.mean_obs_events, 1),
               "freq_obs": jnum(r.freq_obs_median, 3), "freq_prd": jnum(r.freq_prd_median, 3)}
        for m in THR_METRICS:
            rec[m] = jnum(r[f"{m}_median"], 3)
            rec[f"{m}_lo"] = jnum(r[f"{m}_p25"], 3)
            rec[f"{m}_hi"] = jnum(r[f"{m}_p75"], 3)
        rows.append(rec)
    threshold_curves[skey] = rows
    print(f"  threshold_curves[{skey}]: {len(rows)} thresholds")
write_json("threshold_curves.json", threshold_curves)

# ------------------ 7. reproducibility + application classes (curated thesis)
# Transcribed from the verified thesis tables: Table 4.5 (Colab runtime) and the
# application-classes table in Chapter 5. Links verified against the thesis.
repro = {
    "runtime": {
        "total": 72.1,
        "rows": [
            {"nb": "02", "stage": "LSEQM+DL bias correction (per-dekad CNN training)", "min": 16.3},
            {"nb": "03", "stage": "Verification metrics (31-metric catalogue)", "min": 45.4},
            {"nb": "04", "stage": "Quality assessment (CQI sub-scores)", "min": 6.9},
            {"nb": "05", "stage": "Station validation against BMKG", "min": 0.5},
            {"nb": "06", "stage": "Visualisation", "min": 3.0},
        ],
        "env": "Python 3.12 · TensorFlow 2.20.0 · NumPy 2.0.2 · netCDF4 1.7.4",
        "domain": "Bali subdomain · 80 of 126 land pixels corrected · 24 of 24 CPC cells fitted · all 36 dekads",
        "hardware": "Free-tier Google Colab CPU · no hardware accelerator",
    },
    "served": {
        "true":  {"label": "Well-served", "skill": "Distributional properties",
                  "metrics": "SDR · KS p · Q95/Q99 ratios · wet-day frequency"},
        "false": {"label": "Poorly-served", "skill": "Day-by-day timing",
                  "metrics": "Pearson r · RMSE · NSE (bounded by raw satellite)"},
    },
    # served flag tracks whether the application needs the daily distribution
    # (yes = well-served) or day-specific timing (no); topic repeats on both sides.
    "applications": [
        {"name": "SPI / SPEI drought monitoring", "topic": "Drought", "served": True},
        {"name": "Flood-frequency analysis (multi-year record)", "topic": "Flood", "served": True},
        {"name": "Threshold-exceedance climatology", "topic": "Climate", "served": True},
        {"name": "Hydrological calibration vs multi-year statistics", "topic": "Hydrology", "served": True},
        {"name": "Climate-risk insurance pricing", "topic": "Risk", "served": True},
        {"name": "Agricultural dekadal water balance", "topic": "Agriculture", "served": True},
        {"name": "Real-time flood-event nowcasting", "topic": "Flood", "served": False},
        {"name": "Single-event / day-specific forecasting", "topic": "Climate", "served": False},
        {"name": "Day-1 hydrological model initialisation", "topic": "Hydrology", "served": False},
        {"name": "Day-specific agricultural field operations", "topic": "Agriculture", "served": False},
    ],
    "links": {
        "github": "https://github.com/bennyistanto/hybrid-bias-correction",
        "zenodo": "https://doi.org/10.5281/zenodo.20287847",
        "zenodo_doi": "10.5281/zenodo.20287847",
    },
}
write_json("repro.json", repro, indent=2)

# --------------------- 8. sensitivity sweep (curated thesis, verbatim) ------
# Bali sweep, pooled over 36 dekads. Values transcribed verbatim from
# docs/technical/sensitivity-analysis.qmd (same source as the thesis figure);
# defaults match config.py / config.yml (alpha 0.70, GPD 80th, saturation 2).
sensitivity = {
    "envelope": [0.332, 0.348],
    "span": 0.016,
    "note": "Bali subdomain, pooled per-pixel metrics over 36 dekads",
    "metrics": [
        {"key": "r", "label": "Pearson r", "goal": "high"},
        {"key": "rb", "label": "Relative bias (%)", "goal": "zero"},
        {"key": "sdr", "label": "SDR", "goal": "one"},
        {"key": "rmse", "label": "RMSE (mm/day)", "goal": "low"},
        {"key": "nse", "label": "NSE", "goal": "high"},
    ],
    "params": [
        {"key": "alpha", "label": "Blending weight α", "unit": "", "default": 0.70,
         "values": [0.50, 0.60, 0.70, 0.80, 0.90],
         "r":    [0.344, 0.343, 0.340, 0.336, 0.332],
         "rb":   [-14.5, -14.9, -15.0, -15.2, -15.5],
         "sdr":  [0.910, 0.907, 0.905, 0.903, 0.900],
         "rmse": [5.88, 5.89, 5.89, 5.90, 5.91],
         "nse":  [-0.40, -0.40, -0.41, -0.41, -0.41]},
        {"key": "gpd", "label": "GPD threshold percentile", "unit": "th pct", "default": 80,
         "values": [70, 75, 80, 85, 90],
         "r":    [0.345, 0.342, 0.340, 0.337, 0.335],
         "rb":   [-13.0, -13.7, -15.0, -16.2, -18.0],
         "sdr":  [0.923, 0.919, 0.905, 0.891, 0.867],
         "rmse": [5.96, 5.96, 5.89, 5.82, 5.72],
         "nse":  [-0.42, -0.41, -0.41, -0.40, -0.39]},
        {"key": "density", "label": "Density saturation count", "unit": "stations", "default": 2,
         "values": [1, 2, 3, 4, 5],
         "r":    [0.348, 0.340, 0.336, 0.334, 0.333],
         "rb":   [-14.7, -15.0, -15.3, -15.3, -15.4],
         "sdr":  [0.911, 0.905, 0.903, 0.902, 0.901],
         "rmse": [5.88, 5.89, 5.90, 5.90, 5.91],
         "nse":  [-0.40, -0.41, -0.41, -0.41, -0.41]},
    ],
}
write_json("sensitivity.json", sensitivity, indent=2)

# ------------------- 9. paths forward (curated thesis Table 5.x) ------------
# Four routes to raise the temporal ceiling, from the Chapter 5 paths-forward
# table. distance_ord / ready are ordinal positions for the map (1 low - 3 high);
# only the local-day path has a quantified lift.
paths = {
    "recommended": "localday",
    "paths": [
        {"key": "multivariate", "name": "Multivariate QM",
         "distance": "Moderate", "distance_ord": 2, "payoff": 1, "lift": None,
         "cost": "Larger calibration samples; tens of times more compute",
         "effect": "Open question - depends on the joint structure of the reference",
         "summary": "Correct the joint distribution across pixels and time lags, not each margin alone. Not bound by the rank-preservation argument, but the recoverable joint structure is limited by a 0.5 deg, 172-station reference."},
        {"key": "subdaily", "name": "Sub-daily disaggregation",
         "distance": "Moderate", "distance_ord": 2, "payoff": 2, "lift": None,
         "cost": "Half-hourly archive + geostationary IR (Himawari-9)",
         "effect": "Moderate - recovers within-day timing only",
         "summary": "Redistribute the corrected daily total across the day via cloud-tracking on 10-minute Himawari-9 imagery. Keeps the daily distribution and adds a within-day timing signal on top."},
        {"key": "stochastic", "name": "Stochastic disaggregation",
         "distance": "Large", "distance_ord": 3, "payoff": 1, "lift": None,
         "cost": "Ensemble verification stack on the downstream side",
         "effect": "Reframes timing as uncertainty rather than removing it",
         "summary": "Sample an ensemble of plausible daily timings consistent with the corrected distribution. Scoring shifts to CRPS and rank histograms. Best for users who are already ensemble-aware."},
        {"key": "localday", "name": "Local-day re-aggregation",
         "distance": "Small", "distance_ord": 1, "payoff": 3, "lift": [0.34, 0.57],
         "cost": "Re-run the pipeline on a re-aggregated archive (modest, archive-intensive)",
         "effect": "Targets the UTC-vs-local-day artefact; most actionable for UTC+7 to UTC+9",
         "summary": "Re-aggregate native half-hourly IMERG to a 24-hour window ending 07:00 local before correcting. Removes the calendar mismatch with no methodology change; expected lift r 0.34 to 0.57. The path the reproducible pipeline most directly enables."},
    ],
}
write_json("paths.json", paths, indent=2)

# ---------------- 10. Taylor statistics: 6 products, per-station + pooled -----
# From the authoritative Taylor table (the source of thesis Fig 4.6). Each product
# is scored vs the BMKG series; SDR = std_prd / std_obs (std_obs is the same BMKG
# reference across products at a station). Whole-record stats, not dekad averages.
TPRODUCTS = [("cpc", "CPC-UNI"), ("imergl", "IMERG-L (raw)"), ("imergf", "IMERG-F"),
             ("ls", "LS"), ("lseqm", "LSEQM"), ("lseqmdl", "LSEQM+DL")]
tdf = pd.read_csv(ROOT / "data/output/figures/taylor/taylor_statistics_per_station.csv")
t_stations = []
for r in tdf.itertuples():
    prod = {}
    for key, _ in TPRODUCTS:
        corr = getattr(r, f"{key}_correlation")
        so, sp = getattr(r, f"{key}_std_obs"), getattr(r, f"{key}_std_prd")
        sdr = sp / so if (np.isfinite(so) and so != 0) else float("nan")
        prod[key] = {"r": jnum(corr, 3), "sdr": jnum(sdr, 3),
                     "rmse": jnum(getattr(r, f"{key}_rmse"), 2),
                     "bias": jnum(getattr(r, f"{key}_bias"), 3)}
    t_stations.append({"id": int(r.station_id), "name": str(r.station_name),
                       "region": str(r.region), "lon": round(float(r.lon), 3),
                       "lat": round(float(r.lat), 3), "prod": prod})
pooled = {}
for key, _ in TPRODUCTS:
    grab = lambda m: [s["prod"][key][m] for s in t_stations if s["prod"][key][m] is not None]
    pooled[key] = {"r": round(float(np.mean(grab("r"))), 3), "sdr": round(float(np.mean(grab("sdr"))), 3),
                   "rmse": round(float(np.mean(grab("rmse"))), 2), "bias": round(float(np.mean(grab("bias"))), 3)}
write_json("taylor.json", {"products": [{"key": k, "label": l} for k, l in TPRODUCTS],
                           "stations": t_stations, "pooled": pooled})
print(f"  taylor: {len(t_stations)} stations x {len(TPRODUCTS)} products; "
      f"pooled r lseqmdl={pooled['lseqmdl']['r']} imergl={pooled['imergl']['r']}")

# -------------- 11. seasonal stability: monthly, 4 metrics, per stage (Fig 4.10)
# EXACT thesis method (fig_04_temporal_stability.py): the gridded metricssd_cpc
# fields (in-sample vs CPC-UNI), per-pixel spatial median in each dekad, then the
# mean of the three dekads in each calendar month. NOT the station validation -
# that inverts the KS panel because the corrected stages pass KS vs BMKG but fail
# it vs CPC.
SEASON_VARS = ["stdev_ratio", "rmse", "ks_pvalue", "csi"]


def _smed(lst, nd, scale=1.0):
    return jnum(np.mean(lst) * scale, nd) if lst else None


seasonal = {}
for skey in ["ls", "lseqm", "lseqmdl"]:
    rows = []
    for month in range(1, 13):
        acc = {v: [] for v in SEASON_VARS}
        for dd in ("01", "11", "21"):
            fp = ROOT / f"data/output/metrics_{skey}/idn_cli_metricssd_cpc_imergl_{skey}_month{month:02d}_dekad{dd}.nc4"
            if not fp.exists():
                continue
            ds = xr.open_dataset(fp, decode_timedelta=False)
            for v in SEASON_VARS:
                if v in ds:
                    a = ds[v].values.ravel()
                    a = a[np.isfinite(a)]
                    if a.size:
                        acc[v].append(float(np.median(a)))
        rows.append({"month": month, "sdr": _smed(acc["stdev_ratio"], 3),
                     "rmse": _smed(acc["rmse"], 2), "ks": _smed(acc["ks_pvalue"], 2, 100.0),
                     "csi": _smed(acc["csi"], 3)})
    seasonal[skey] = rows
write_json("seasonal.json", seasonal)
print(f"  seasonal: gridded vs CPC; lseqmdl SDR Jan={seasonal['lseqmdl'][0]['sdr']} "
      f"KS Jan ls={seasonal['ls'][0]['ks']} lseqmdl={seasonal['lseqmdl'][0]['ks']}")

# -------- 12. monthly Taylor (Fig 5.2): per-month, per-product aggregate ------
# Exact thesis method (taylor_helpers.compute_period_stats): per station, median
# of the month's dekads (n >= 30); the product point is the median across stations.
tdek = pd.read_csv(ROOT / "data/output/figures/taylor/taylor_statistics_per_dekad.csv")
MIN_N = 30
MPROD = [("cpc", "CPC-UNI", "#8c564b"), ("imergl", "IMERG-L", "#1f77b4"), ("imergf", "IMERG-F", "#17becf"),
         ("ls", "LS", "#2ca02c"), ("lseqm", "LSEQM", "#ff7f0e"), ("lseqmdl", "LSEQM+DL", "#d62728")]


def _month_taylor(months):
    sub = tdek[tdek["month"].isin(months)]
    out = {}
    cloud = []  # every per-station (r, sdr) point, all products pooled, for the spread cloud
    for key, _, _ in MPROD:
        cc, so, sp, nn = f"{key}_correlation", f"{key}_std_obs", f"{key}_std_prd", f"{key}_n"
        rs, sds = [], []
        for _, ss in sub.groupby("station_id"):
            v = ss[ss[cc].notna() & ss[so].notna() & (ss[so] > 0) & ss[sp].notna() & (ss[nn] >= MIN_N)]
            if len(v):
                mc, mr = v[cc].median(), (v[sp] / v[so]).median()
                if np.isfinite(mc) and np.isfinite(mr):
                    rs.append(mc); sds.append(mr)
                    cloud.append([jnum(mc, 3), jnum(mr, 3)])
        out[key] = {"r": jnum(np.median(rs), 3) if rs else None,
                    "sdr": jnum(np.median(sds), 3) if sds else None, "n": len(rs)}
    out["cloud"] = cloud
    return out


monthly_taylor = {"products": [{"key": k, "label": l, "color": c} for k, l, c in MPROD],
                  "months": {str(m): _month_taylor({m}) for m in range(1, 13)}}
monthly_taylor["months"]["all"] = _month_taylor(set(range(1, 13)))
write_json("monthly_taylor.json", monthly_taylor)
print(f"  monthly_taylor: 12 months + all; all-year lseqmdl "
      f"r={monthly_taylor['months']['all']['lseqmdl']['r']} sdr={monthly_taylor['months']['all']['lseqmdl']['sdr']}")

print("Done. Files in", OUT)
