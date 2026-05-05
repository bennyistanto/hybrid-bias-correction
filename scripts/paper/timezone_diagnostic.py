"""
timezone_diagnostic — day-shift lag tests (B1) and CPC convention check (B3).

Background
----------
All pairing in `src/` is naive calendar-date intersection. BMKG is local civil
day (WIB/WITA/WIT, UTC+7..+9), IMERG is 00Z-00Z UTC, CPC-UNI is a 12Z-12Z
gauge day with an unverified label convention. This module quantifies the
resulting timing penalty by sweeping integer day shifts and reporting the
shift that maximizes Pearson r.

This is paper-only diagnostic; nothing in `src/` is modified.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

from .paper_helpers import (
    ALL_PERIODS, load_corrected_da, load_reference_da,
    load_station_obs, load_station_locations_df,
)
from src.metrics import unify_cpc_for_metrics
from src import config


# +++++++++++++++++++++++++++++++++++++++++
# B1 — Station day-shift lag test
# +++++++++++++++++++++++++++++++++++++++++

def day_shift_lag_test_station(method='lseqmdl', shifts=(-1, 0, 1),
                               min_valid_days=60):
    """
    For each BMKG station, pair daily obs against the corrected gridded value
    at the nearest cell using integer day shifts and report the shift that
    maximizes Pearson r.

    Parameters
    ----------
    method : {'ls', 'lseqm', 'lseqmdl'}
    shifts : iterable of int
        Day shifts (in days) to apply to the gridded series before pairing.
        Positive shift = grid time + shift days.
    min_valid_days : int
        Minimum paired non-missing days for a station/shift to be reported.

    Returns
    -------
    pandas.DataFrame
        Index = station WMO ID, columns = one r per shift, plus 'best_shift'.
    """
    logging.info("Loading station observations and locations...")
    obs_df = load_station_obs()
    locs = load_station_locations_df()

    logging.info(f"Loading and concatenating {method} corrected fields...")
    pieces = []
    for month, dk in ALL_PERIODS:
        da = load_corrected_da(method, month, dk)
        if da is not None:
            pieces.append(da)
    test = xr.concat(pieces, dim='time').sortby('time')
    grid_time = pd.DatetimeIndex(test.time.values)

    rows = []
    for _, row in locs.iterrows():
        wmo = int(row['ID_WMO'])
        if wmo not in obs_df.columns:
            continue
        ts = test.sel(lat=float(row['Lat']), lon=float(row['Lon']),
                      method='nearest').values
        grid_ts = pd.Series(ts, index=grid_time)
        obs_ts = obs_df[wmo]

        out = {'station_id': wmo}
        rs = {}
        for s in shifts:
            shifted = grid_ts.copy()
            shifted.index = shifted.index + pd.Timedelta(days=s)
            common = obs_ts.dropna().index.intersection(shifted.dropna().index)
            if len(common) < min_valid_days:
                out[f'r_shift{s:+d}'] = np.nan
                rs[s] = -np.inf
                continue
            a = obs_ts.loc[common].values.astype(np.float64)
            b = shifted.loc[common].values.astype(np.float64)
            if np.std(a) == 0 or np.std(b) == 0:
                out[f'r_shift{s:+d}'] = np.nan
                rs[s] = -np.inf
                continue
            r, _ = pearsonr(a, b)
            out[f'r_shift{s:+d}'] = r
            rs[s] = r
        finite = {k: v for k, v in rs.items() if np.isfinite(v)}
        out['best_shift'] = max(finite, key=finite.get) if finite else np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index('station_id')


# +++++++++++++++++++++++++++++++++++++++++
# B1 — Grid-vs-grid day-shift lag test
# +++++++++++++++++++++++++++++++++++++++++

def day_shift_lag_test_grid(method='lseqmdl', ref_label='cpc',
                            shifts=(-1, 0, 1), sample_pixels=200,
                            seed=0):
    """
    Sample land pixels from the corrected field and run a day-shift lag test
    against the reference. Returns one r per (pixel, shift) and the best shift
    per pixel.

    Parameters
    ----------
    method : {'ls', 'lseqm', 'lseqmdl'}
    ref_label : {'cpc', 'imergl', 'imergf'}
    shifts : iterable of int
    sample_pixels : int
        Number of land pixels to randomly sample (full-domain is expensive).
    seed : int

    Returns
    -------
    pandas.DataFrame
        Index = pixel id, columns = r per shift + 'best_shift', 'lat', 'lon'.
    """
    from src.utility import apply_land_sea_mask

    ref = load_reference_da(ref_label)
    pieces = []
    for month, dk in ALL_PERIODS:
        da = load_corrected_da(method, month, dk)
        if da is not None:
            pieces.append(da)
    test = xr.concat(pieces, dim='time').sortby('time')

    if ref_label == 'cpc':
        ref_da = unify_cpc_for_metrics(
            ref.to_dataset(name=config.CPC_PRECIP_VAR),
            test.to_dataset(name=config.IMERG_PRECIP_VAR),
        )[config.CPC_PRECIP_VAR]
    else:
        ref_da = ref.interp(lat=test.lat, lon=test.lon, method='nearest')

    ref_da, test = xr.align(ref_da, test, join='inner')
    ref_da = apply_land_sea_mask(ref_da, config.mask_file)
    test = apply_land_sea_mask(test, config.mask_file)

    # Pick land pixels: any cell where the first ref slice is not NaN
    first = ref_da.isel(time=0).values
    ii, jj = np.where(np.isfinite(first))
    rng = np.random.default_rng(seed)
    if len(ii) > sample_pixels:
        sel = rng.choice(len(ii), size=sample_pixels, replace=False)
        ii, jj = ii[sel], jj[sel]

    ref_vals = ref_da.values  # (t, lat, lon)
    test_vals = test.values
    times = pd.DatetimeIndex(ref_da.time.values)
    lats = ref_da.lat.values
    lons = ref_da.lon.values

    rows = []
    for k, (i, j) in enumerate(zip(ii, jj)):
        a_full = pd.Series(ref_vals[:, i, j], index=times)
        b_full = pd.Series(test_vals[:, i, j], index=times)
        out = {'pixel': k, 'lat': float(lats[i]), 'lon': float(lons[j])}
        rs = {}
        for s in shifts:
            shifted = b_full.copy()
            shifted.index = shifted.index + pd.Timedelta(days=s)
            common = a_full.dropna().index.intersection(shifted.dropna().index)
            if len(common) < 60:
                out[f'r_shift{s:+d}'] = np.nan
                rs[s] = -np.inf
                continue
            a = a_full.loc[common].values
            b = shifted.loc[common].values
            if np.std(a) == 0 or np.std(b) == 0:
                out[f'r_shift{s:+d}'] = np.nan
                rs[s] = -np.inf
                continue
            r, _ = pearsonr(a, b)
            out[f'r_shift{s:+d}'] = r
            rs[s] = r
        finite = {k_: v for k_, v in rs.items() if np.isfinite(v)}
        out['best_shift'] = max(finite, key=finite.get) if finite else np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index('pixel')


# +++++++++++++++++++++++++++++++++++++++++
# B3 — CPC day-label convention check
# +++++++++++++++++++++++++++++++++++++++++

def cpc_imerg_convention_test(shifts=(-1, 0, 1), sample_pixels=200, seed=0):
    """
    Empirically determine the CPC-UNI day-label convention by lag-testing
    raw CPC against raw IMERG-L (no correction).

    A best_shift dominated by 0 means CPC date D ≈ IMERG date D (start-date
    convention or close enough). A best_shift dominated by -1 or +1 means
    the labels are offset by one day.

    Parameters
    ----------
    shifts : iterable of int
    sample_pixels : int
    seed : int

    Returns
    -------
    pandas.DataFrame
        Per-pixel r per shift, plus 'best_shift', 'lat', 'lon'.
    pandas.Series
        Distribution of best_shift values.
    """
    from src.utility import apply_land_sea_mask

    cpc = load_reference_da('cpc')
    imergl = load_reference_da('imergl')

    cpc_ds = unify_cpc_for_metrics(
        cpc.to_dataset(name=config.CPC_PRECIP_VAR),
        imergl.to_dataset(name=config.IMERG_PRECIP_VAR),
    )
    cpc_da = cpc_ds[config.CPC_PRECIP_VAR]
    imerg_da = imergl.interp(lat=cpc_da.lat, lon=cpc_da.lon, method='nearest')
    cpc_da, imerg_da = xr.align(cpc_da, imerg_da, join='inner')

    cpc_da = apply_land_sea_mask(cpc_da, config.mask_file)
    imerg_da = apply_land_sea_mask(imerg_da, config.mask_file)

    first = cpc_da.isel(time=0).values
    ii, jj = np.where(np.isfinite(first))
    rng = np.random.default_rng(seed)
    if len(ii) > sample_pixels:
        sel = rng.choice(len(ii), size=sample_pixels, replace=False)
        ii, jj = ii[sel], jj[sel]

    a_vals = cpc_da.values
    b_vals = imerg_da.values
    times = pd.DatetimeIndex(cpc_da.time.values)
    lats = cpc_da.lat.values
    lons = cpc_da.lon.values

    rows = []
    for k, (i, j) in enumerate(zip(ii, jj)):
        a_full = pd.Series(a_vals[:, i, j], index=times)
        b_full = pd.Series(b_vals[:, i, j], index=times)
        out = {'pixel': k, 'lat': float(lats[i]), 'lon': float(lons[j])}
        rs = {}
        for s in shifts:
            shifted = b_full.copy()
            shifted.index = shifted.index + pd.Timedelta(days=s)
            common = a_full.dropna().index.intersection(shifted.dropna().index)
            if len(common) < 60:
                out[f'r_shift{s:+d}'] = np.nan
                rs[s] = -np.inf
                continue
            a = a_full.loc[common].values
            b = shifted.loc[common].values
            if np.std(a) == 0 or np.std(b) == 0:
                out[f'r_shift{s:+d}'] = np.nan
                rs[s] = -np.inf
                continue
            r, _ = pearsonr(a, b)
            out[f'r_shift{s:+d}'] = r
            rs[s] = r
        finite = {k_: v for k_, v in rs.items() if np.isfinite(v)}
        out['best_shift'] = max(finite, key=finite.get) if finite else np.nan
        rows.append(out)

    df = pd.DataFrame(rows).set_index('pixel')
    dist = df['best_shift'].value_counts().sort_index()
    return df, dist
