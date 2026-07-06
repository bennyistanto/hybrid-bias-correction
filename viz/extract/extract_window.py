"""Extract the window-offset diagnostics (thesis Figs 4.4-4.6) for the viz.

Reuses the thesis figures' own helpers (subdaily_helpers) on the sufficient-
statistics sweeps, so the viz cannot drift from the paper:
  - era_window.json : per-year UTC-day vs best-window r and peak offset (Fig 4.4)
  - window_gpm.json : GPM-era r(h) by timezone band with IQR + season-invariant
                      h*, and per-station h* / lift (Figs 4.5, 4.6)

    python viz/extract/extract_window.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

ROOT = Path(r"C:\Users\benny\OneDrive\Documents\Github\hybrid-bias-correction")
sys.path.insert(0, str(ROOT / "paper" / "thesis" / "scripts"))
from subdaily_helpers import (SUBDAILY, HOUR_SHIFTS, r_from_stats, r_at, peak_offset,  # noqa: E402
                              season_stats, pool_stations, per_station_hstar_peak, SEASON_ORDER,
                              reduce_cells, station_cells, clip_to_land_01)

OUT = ROOT / "viz" / "src" / "data"
MAPS = OUT / "maps"
MAPS.mkdir(parents=True, exist_ok=True)
H = HOUR_SHIFTS
BANDS = [(7, "WIB (UTC+7)"), (8, "WITA (UTC+8)"), (9, "WIT (UTC+9)")]


def rnd(a, nd):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(a)]


# ---------------------------------------- Fig 4.4: per-year era reveal --------
FULL = np.load(SUBDAILY / "subdaily_seasonal_results_2001_2021.npz", allow_pickle=True)
By = FULL["stats_year_month"].sum(axis=1)
years = [int(y) for y in FULL["years"]]
r0, pk, hstar, hlo, hhi = [], [], [], [], []
for i in range(len(years)):
    rc = r_from_stats(By[i])
    r0.append(r_at(rc, 0))
    ho, pr = peak_offset(rc)
    hstar.append(ho); pk.append(pr)
    near = H[np.isfinite(rc) & (rc >= np.nanmax(rc) - 0.02)]
    hlo.append(int(near.min())); hhi.append(int(near.max()))
(OUT / "era_window.json").write_text(json.dumps({
    "years": years, "r0": rnd(r0, 3), "pk": rnd(pk, 3), "hstar": rnd(hstar, 1),
    "hlo": hlo, "hhi": hhi, "gpm_split": 2014.5, "convention_offset": -23}), encoding="utf-8")
print(f"  era_window: {years[0]}-{years[-1]}")

# ----------------------- Figs 4.5 / 4.6: GPM-era band + station diagnostics ---
GPM = np.load(SUBDAILY / "subdaily_seasonal_results_2015_2021.npz", allow_pickle=True)
A = GPM["stats_month"]                       # (12, n_st, n_h, 6)
tz, lon, lat, wmo = GPM["timezone"], GPM["lon"], GPM["lat"], GPM["wmo"]
# station names by WMO id (stations.json is written by extract_core)
name_by_wmo = {int(s["wmo"]): s["name"]
               for s in json.loads((OUT / "stations.json").read_text(encoding="utf-8"))}

# 4.5a: GPM-era r(h) by band, mean + IQR across stations
rps = r_from_stats(A.sum(axis=0))            # (n_st, n_h)
rh_bands = []
for z, name in BANDS:
    sel = tz == z
    mr = np.nanmean(rps[sel], axis=0)
    pi = int(np.nanargmax(mr))
    rh_bands.append({"tz": int(z), "name": name, "n": int(sel.sum()),
                     "mean": rnd(mr, 3),
                     "q25": rnd(np.nanpercentile(rps[sel], 25, axis=0), 3),
                     "q75": rnd(np.nanpercentile(rps[sel], 75, axis=0), 3),
                     "peakH": int(H[pi]), "peakR": round(float(mr[pi]), 3)})

# 4.5b: h*(season) - pooled + per-band median across the 12 running seasons
pooled_h, band_h = [], {7: [], 8: [], 9: []}
for s in SEASON_ORDER:
    sm = season_stats(A, s)
    pooled_h.append(peak_offset(r_from_stats(pool_stations(sm)))[0])
    hs_s, _, _, val_s = per_station_hstar_peak(sm)
    for z in (7, 8, 9):
        band_h[z].append(float(np.nanmedian(hs_s[(tz == z) & val_s])))

# 4.6: per-station h*, r0, peak r, lift
hs, pr_st, r0_st, valid = per_station_hstar_peak(A.sum(axis=0))
stations = [{"name": name_by_wmo.get(int(wmo[i]), str(int(wmo[i]))),
             "lon": round(float(lon[i]), 3), "lat": round(float(lat[i]), 3), "tz": int(tz[i]),
             "hstar": int(hs[i]), "r0": round(float(r0_st[i]), 3),
             "peakr": round(float(pr_st[i]), 3), "lift": round(float(pr_st[i] - r0_st[i]), 3)}
            for i in range(len(lon)) if valid[i]]

window_gpm = {
    "h": [int(x) for x in H],
    "bands": rh_bands,
    "season": {"order": list(SEASON_ORDER), "pooled": rnd(pooled_h, 1),
               "bands": {str(z): rnd(band_h[z], 1) for z in (7, 8, 9)}},
    "stations": stations,
    "summary": {"median_hstar": int(np.nanmedian(hs)),
                "iqr_hstar": [int(np.nanpercentile(hs, 25)), int(np.nanpercentile(hs, 75))],
                "median_lift": round(float(np.nanmedian(pr_st - r0_st)), 3),
                "median_peakr": round(float(np.nanmedian(pr_st)), 3)},
    "convention_offset": -23,
    # daily-vs-monthly table transcribed verbatim from thesis Table (GPM era):
    # a one-day window shift is invisible at the monthly scale.
    "daily_monthly": {
        "n_station_months": 5338,
        "rows": [
            {"window": "UTC day (h = 0)", "h": 0, "daily": 0.20, "monthly": 0.80},
            {"window": "Gauge-matched (h = −23)", "h": -23, "daily": 0.57, "monthly": 0.81},
        ],
    },
}
(OUT / "window_gpm.json").write_text(json.dumps(window_gpm), encoding="utf-8")
print(f"  window_gpm: {len(stations)} stations; median h*={window_gpm['summary']['median_hstar']} "
      f"IQR={window_gpm['summary']['iqr_hstar']} median lift={window_gpm['summary']['median_lift']}")

# ------------------- Fig 5.4: gridded whole-domain window offset --------------
# Full-record h* maps (native vs harmonised, pooled 2001-2014 + 2015-2025) and the
# pooled r(h) over gauged cells vs the whole domain, per era.
Sn = Sh = clat = clon = None
for t in ("2001_2014", "2015_2025"):
    z = np.load(SUBDAILY / f"gridded_cpc_window_stats_{t}.npz")
    Sn = z["stats_native"] if Sn is None else Sn + z["stats_native"]
    Sh = z["stats_harmonised"] if Sh is None else Sh + z["stats_harmonised"]
    clat, clon = z["clat"], z["clon"]
hn = reduce_cells(Sn)[0].reshape(len(clat), len(clon))
hh = reduce_cells(Sh)[0].reshape(len(clat), len(clon))

adm0 = gpd.read_file(ROOT / "data/subset/bnd/wld_bnd_adm0.shp", bbox=(93, -13, 143, 8))


def render_hstar(field, path):
    mlat, mlon, hs01 = clip_to_land_01(field, clat, clon)
    ext = [float(mlon.min()), float(mlon.max()), float(mlat.min()), float(mlat.max())]
    fig = plt.figure(figsize=(9, 9 / ((ext[1] - ext[0]) / (ext[3] - ext[2]))), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    cm = plt.get_cmap("viridis").copy(); cm.set_bad(alpha=0.0)
    ax.imshow(np.ma.masked_invalid(hs01), extent=ext, origin="lower",
              vmin=-26, vmax=2, cmap=cm, interpolation="nearest")
    adm0.boundary.plot(ax=ax, color="#3a3a3a", linewidth=0.4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.savefig(path, transparent=True, dpi=150)
    plt.close(fig)


render_hstar(hn, MAPS / "gridded_hstar_native.png")
render_hstar(hh, MAPS / "gridded_hstar_harmonised.png")

cells = station_cells(clat, clon)


def _curves(tag):
    z = np.load(SUBDAILY / f"gridded_cpc_window_stats_{tag}.npz")
    o = {}
    for pair in ("native", "harmonised"):
        S = z[f"stats_{pair}"]
        o[f"{pair}_gauged"] = r_from_stats(S[cells].sum(0))
        o[f"{pair}_whole"] = r_from_stats(S.sum(0))
    return o


gpm, trmm = _curves("2015_2021"), _curves("2001_2014")
CURVES = [
    ("native, gauged, GPM",      "#111111", False, 2.6, gpm["native_gauged"]),
    ("native, gauged, TRMM",     "#111111", True,  2.0, trmm["native_gauged"]),
    ("native, whole, GPM",       "#9a9a9a", False, 1.5, gpm["native_whole"]),
    ("native, whole, TRMM",      "#9a9a9a", True,  1.3, trmm["native_whole"]),
    ("harmonised, gauged, GPM",  "#882255", False, 2.6, gpm["harmonised_gauged"]),
    ("harmonised, gauged, TRMM", "#882255", True,  2.0, trmm["harmonised_gauged"]),
]
window_gridded = {
    "h": [int(x) for x in H],
    "curves": [{"label": lab, "color": col, "dash": dash, "width": w, "r": rnd(r, 3)}
               for lab, col, dash, w, r in CURVES],
    "summary": {"gauged_peak": round(float(np.nanmax(gpm["harmonised_gauged"])), 3),
                "whole_peak": round(float(np.nanmax(gpm["native_whole"])), 3)},
}
(OUT / "window_gridded.json").write_text(json.dumps(window_gridded), encoding="utf-8")
print(f"  window_gridded: 2 h* maps + {len(CURVES)} r(h) curves; "
      f"gauged peak={window_gridded['summary']['gauged_peak']} whole peak={window_gridded['summary']['whole_peak']}")
