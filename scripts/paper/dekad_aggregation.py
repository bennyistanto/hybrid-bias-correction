"""
dekad_aggregation — temporal-scale-aware metrics for the paper.

Background
----------
The package-level metrics in `src/metrics.py` compute Pearson r, RMSE, NSE on
*daily paired values* within each dekad. For tropical IMERG vs gauge-interpolated
references, daily-scale correlations are bounded above by ~0.3-0.4 due to:

1. Sub-daily satellite retrieval timing errors
2. Misalignment of accumulation windows: IMERG (00Z-00Z UTC),
   CPC-UNI (12Z-12Z gauge day), BMKG stations (local civil day, WIB/WITA/WIT)
3. Point-to-grid representativeness gap at 0.1 deg

This module computes metrics on *dekad-aggregated totals* instead — one value
per dekad per pixel/station, time series of (n_years * 36 dekads). At the
dekad scale, the timezone offset of 7-9 hours becomes a ~3% leakage instead
of ~30%, and r typically jumps from 0.3 to 0.6-0.8 in the literature
(Beck et al. 2019; Sun et al. 2018; Tan et al. 2016).

This is paper-only post-hoc analysis; the package output remains unchanged.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr

from .paper_helpers import (
    METHODS, ALL_PERIODS, DEKAD_MAP,
    load_corrected_da, load_reference_da, load_station_obs,
    load_station_locations_df, dekad_index_for_date,
)
from src.metrics import compute_pixel_metrics, slice_month_dekad, unify_cpc_for_metrics
from src.utility import apply_land_sea_mask
from src import config


# +++++++++++++++++++++++++++++++++++++++++
# Core dekad aggregation
# +++++++++++++++++++++++++++++++++++++++++

def aggregate_to_dekad_totals(da):
    """
    Sum a daily DataArray into dekad totals.

    Parameters
    ----------
    da : xarray.DataArray
        Daily precipitation with a 'time' dimension. May have lat/lon or be 1D.

    Returns
    -------
    xarray.DataArray
        Dekad totals with 'time' dimension where each entry is one dekad.
        Time index uses the first day of each dekad (day 1, 11, or 21).
        Sum requires at least 5 valid days per dekad; otherwise NaN.
    """
    time = pd.DatetimeIndex(da.time.values)

    # Build dekad ID for each daily timestep: (year, month, dekad_idx)
    dekad_ids = []
    for t in time:
        m = t.month
        d = t.day
        if d <= 10:
            dk = 1
            day_label = 1
        elif d <= 20:
            dk = 2
            day_label = 11
        else:
            dk = 3
            day_label = 21
        dekad_ids.append(pd.Timestamp(year=t.year, month=m, day=day_label))

    dekad_da = da.assign_coords(dekad_id=('time', dekad_ids))

    # Group + sum, requiring at least 5 valid days per dekad
    grouped = dekad_da.groupby('dekad_id')
    totals = grouped.sum(skipna=True, min_count=5)
    totals = totals.rename({'dekad_id': 'time'})
    totals = totals.sortby('time')
    return totals


def aggregate_station_df_to_dekad(station_df):
    """
    Sum a station observations DataFrame to dekad totals.

    Parameters
    ----------
    station_df : pandas.DataFrame
        Daily observations indexed by datetime, columns are station IDs.

    Returns
    -------
    pandas.DataFrame
        Dekad totals indexed by the first day of each dekad.
        Requires at least 5 valid days per dekad; otherwise NaN.
    """
    df = station_df.copy()
    dekad_starts = []
    for t in df.index:
        if t.day <= 10:
            day_label = 1
        elif t.day <= 20:
            day_label = 11
        else:
            day_label = 21
        dekad_starts.append(pd.Timestamp(year=t.year, month=t.month, day=day_label))
    df['_dekad'] = dekad_starts

    # Sum per dekad with min_count=5
    valid_count = df.drop(columns='_dekad').groupby(df['_dekad']).count()
    totals = df.drop(columns='_dekad').groupby(df['_dekad']).sum(min_count=1)
    totals = totals.where(valid_count >= 5)
    totals.index.name = 'time'
    return totals


# +++++++++++++++++++++++++++++++++++++++++
# Grid-vs-grid dekad metrics
# +++++++++++++++++++++++++++++++++++++++++

def compute_dekad_grid_metrics(method='lseqmdl', ref_label='cpc',
                               apply_mask=True, threshold=1.0):
    """
    Compute pixel-level metrics on dekad-aggregated totals.

    For each pixel, builds the time series of dekad totals (25 yrs * 36
    dekads = 900 entries) for both reference and test, then computes the
    full 31-metric suite via `compute_pixel_metrics`.

    Parameters
    ----------
    method : {'ls', 'lseqm', 'lseqmdl'}
    ref_label : {'cpc', 'imergl', 'imergf'}
    apply_mask : bool
        Apply land-sea mask before metric computation.
    threshold : float
        Wet-day threshold (mm/dekad). Note: at dekad scale, "wet" means
        accumulated >= threshold. Set this much higher than 1.0 if you
        want to compare against a true wet-day count.

    Returns
    -------
    xarray.Dataset
        31 metric variables on (lat, lon).
    """
    from src.metrics import _apply_pixel_metrics

    logging.info(f"Loading reference {ref_label}...")
    ref = load_reference_da(ref_label)

    logging.info(f"Loading {method} corrected fields for all periods...")
    corrected_pieces = []
    for month, dk in ALL_PERIODS:
        da = load_corrected_da(method, month, dk)
        if da is None:
            continue
        corrected_pieces.append(da)
    if not corrected_pieces:
        raise RuntimeError(f"No corrected files found for method {method}")
    test = xr.concat(corrected_pieces, dim='time').sortby('time')

    # Align reference to test grid+time
    if ref_label == 'cpc':
        ref_da = unify_cpc_for_metrics(ref.to_dataset(name=config.CPC_PRECIP_VAR), test.to_dataset(name=config.IMERG_PRECIP_VAR))
        ref_da = ref_da[config.CPC_PRECIP_VAR]
    else:
        ref_da = ref.interp(lat=test.lat, lon=test.lon, method='nearest')

    ref_da, test = xr.align(ref_da, test, join='inner')

    if apply_mask:
        ref_da = apply_land_sea_mask(ref_da, config.mask_file)
        test = apply_land_sea_mask(test, config.mask_file)

    logging.info("Aggregating to dekad totals...")
    ref_dk = aggregate_to_dekad_totals(ref_da)
    test_dk = aggregate_to_dekad_totals(test)
    ref_dk, test_dk = xr.align(ref_dk, test_dk, join='inner')

    logging.info(f"Computing 31 metrics on {len(ref_dk.time)} dekad totals per pixel...")
    metrics_ds = _apply_pixel_metrics(ref_dk, test_dk, threshold=threshold)
    return metrics_ds


# +++++++++++++++++++++++++++++++++++++++++
# Station-vs-grid dekad metrics
# +++++++++++++++++++++++++++++++++++++++++

def compute_dekad_station_metrics(method='lseqmdl', threshold=1.0,
                                  min_valid_dekads=10):
    """
    Compute per-station metrics on dekad-aggregated totals.

    Pairs each BMKG station's dekad totals against the corrected gridded
    product extracted at the station's nearest grid cell, then computes
    the full 31-metric suite per station.

    Parameters
    ----------
    method : {'ls', 'lseqm', 'lseqmdl'}
    threshold : float
        Wet-day threshold (mm/dekad). 1.0 keeps the same convention.
    min_valid_dekads : int
        Minimum number of paired non-missing dekad totals to keep a station.

    Returns
    -------
    pandas.DataFrame
        Per-station metrics, indexed by WMO station ID, with the 31 metric
        columns from `METRIC_NAMES` plus 'n_valid_dekads'.
    """
    from src.metrics import METRIC_NAMES

    logging.info("Loading station observations...")
    obs_df = load_station_obs()  # daily, naive datetime index
    station_locs = load_station_locations_df()

    logging.info("Loading and concatenating corrected fields...")
    corrected_pieces = []
    for month, dk in ALL_PERIODS:
        da = load_corrected_da(method, month, dk)
        if da is None:
            continue
        corrected_pieces.append(da)
    test = xr.concat(corrected_pieces, dim='time').sortby('time')

    logging.info("Extracting gridded values at station locations...")
    grid_at_station = {}
    for _, row in station_locs.iterrows():
        wmo = int(row['ID_WMO'])
        ts = test.sel(lat=float(row['Lat']), lon=float(row['Lon']),
                      method='nearest')
        grid_at_station[wmo] = ts.values
    grid_df = pd.DataFrame(
        grid_at_station,
        index=pd.DatetimeIndex(test.time.values),
    )

    logging.info("Aggregating to dekad totals...")
    obs_dk = aggregate_station_df_to_dekad(obs_df)
    grid_dk = aggregate_station_df_to_dekad(grid_df)

    common_stations = sorted(set(obs_dk.columns) & set(grid_dk.columns))
    logging.info(f"Computing dekad metrics for {len(common_stations)} stations...")

    rows = []
    skipped = 0
    for wmo in common_stations:
        obs_ts = obs_dk[wmo]
        grid_ts = grid_dk[wmo]
        common_idx = obs_ts.dropna().index.intersection(grid_ts.dropna().index)
        if len(common_idx) < min_valid_dekads:
            skipped += 1
            continue
        ref_1d = obs_ts.loc[common_idx].values.astype(np.float64)
        test_1d = grid_ts.loc[common_idx].values.astype(np.float64)
        metrics = compute_pixel_metrics(ref_1d, test_1d, threshold)
        row = {'station_id': wmo, 'n_valid_dekads': len(common_idx)}
        for i, name in enumerate(METRIC_NAMES):
            row[name] = metrics[i]
        rows.append(row)

    if skipped:
        logging.info(f"Skipped {skipped} stations with < {min_valid_dekads} paired dekads")
    return pd.DataFrame(rows).set_index('station_id')


# +++++++++++++++++++++++++++++++++++++++++
# B2' — Stratified-by-dekad-of-year metrics (Meaning B)
# +++++++++++++++++++++++++++++++++++++++++
#
# For each pixel (or station) and each of the 36 dekads-of-year, compute
# metrics on the 25-year inter-annual sample. This strips out the monsoon
# seasonal contrast and asks the much harder question: "within just one
# dekad-of-year, across years, does the corrected product track the
# reference's inter-annual variability better than the raw or baseline?"
#
# LS cannot help inter-annual variability within a fixed dekad-of-year
# (it only rescales the mean), so this is exactly the regime where EQM
# and DL refinement are expected to pay off.


def _dekad_of_year_index(time_index):
    """Return 1..36 dekad-of-year label for each timestamp."""
    out = np.empty(len(time_index), dtype=np.int16)
    for i, t in enumerate(time_index):
        if t.day <= 10:
            dk = 0
        elif t.day <= 20:
            dk = 1
        else:
            dk = 2
        out[i] = (t.month - 1) * 3 + dk + 1  # 1..36
    return out


def compute_stratified_dekad_grid_metrics(method='lseqmdl', ref_label='cpc',
                                          apply_mask=True, threshold=1.0):
    """
    Stratified-by-dekad-of-year grid metrics (Meaning B).

    For each pixel and each dekad-of-year (1..36), compute the 31-metric
    suite on the 25-year inter-annual dekad-total sample.

    Returns
    -------
    xarray.Dataset
        31 metric variables on (dekad_of_year, lat, lon). dekad_of_year
        coordinate runs 1..36.
    """
    from src.metrics import _apply_pixel_metrics

    logging.info(f"[stratified] Loading reference {ref_label}...")
    ref = load_reference_da(ref_label)

    logging.info(f"[stratified] Loading {method} corrected fields...")
    corrected_pieces = []
    for month, dk in ALL_PERIODS:
        da = load_corrected_da(method, month, dk)
        if da is not None:
            corrected_pieces.append(da)
    test = xr.concat(corrected_pieces, dim='time').sortby('time')

    if ref_label == 'cpc':
        ref_da = unify_cpc_for_metrics(
            ref.to_dataset(name=config.CPC_PRECIP_VAR),
            test.to_dataset(name=config.IMERG_PRECIP_VAR),
        )[config.CPC_PRECIP_VAR]
    else:
        ref_da = ref.interp(lat=test.lat, lon=test.lon, method='nearest')

    ref_da, test = xr.align(ref_da, test, join='inner')
    if apply_mask:
        ref_da = apply_land_sea_mask(ref_da, config.mask_file)
        test = apply_land_sea_mask(test, config.mask_file)

    logging.info("[stratified] Aggregating to dekad totals...")
    ref_dk = aggregate_to_dekad_totals(ref_da)
    test_dk = aggregate_to_dekad_totals(test)
    ref_dk, test_dk = xr.align(ref_dk, test_dk, join='inner')

    # Stratify by dekad-of-year 1..36
    doy_idx = _dekad_of_year_index(pd.DatetimeIndex(ref_dk.time.values))
    ref_dk = ref_dk.assign_coords(doy=('time', doy_idx))
    test_dk = test_dk.assign_coords(doy=('time', doy_idx))

    per_doy = []
    for d in range(1, 37):
        mask = ref_dk['doy'].values == d
        if mask.sum() < 5:
            logging.warning(f"  doy={d}: only {mask.sum()} samples, skipping")
            continue
        logging.info(f"  doy={d:02d}: {mask.sum()} yearly samples per pixel")
        ref_sub = ref_dk.isel(time=mask)
        test_sub = test_dk.isel(time=mask)
        m = _apply_pixel_metrics(ref_sub, test_sub, threshold=threshold)
        m = m.expand_dims(dekad_of_year=[d])
        per_doy.append(m)

    stacked = xr.concat(per_doy, dim='dekad_of_year')
    return stacked


def compute_stratified_dekad_station_metrics(method='lseqmdl', threshold=1.0,
                                             min_valid_years=5):
    """
    Stratified-by-dekad-of-year station metrics (Meaning B).

    For each station and each dekad-of-year (1..36), compute the 31-metric
    suite on the 25-year inter-annual dekad-total sample paired between the
    station observation and the corrected grid at the station's nearest cell.

    Parameters
    ----------
    min_valid_years : int
        Minimum paired years within a dekad-of-year to keep that (station, doy)
        cell. Lower than min_valid_dekads in the pooled version because each
        cell only has up to 25 samples.

    Returns
    -------
    pandas.DataFrame
        Long-form: one row per (station_id, dekad_of_year), 31 metric columns.
    """
    from src.metrics import METRIC_NAMES

    logging.info("[stratified] Loading station observations...")
    obs_df = load_station_obs()
    station_locs = load_station_locations_df()

    logging.info("[stratified] Loading corrected fields...")
    pieces = []
    for month, dk in ALL_PERIODS:
        da = load_corrected_da(method, month, dk)
        if da is not None:
            pieces.append(da)
    test = xr.concat(pieces, dim='time').sortby('time')

    logging.info("[stratified] Extracting grid at station locations...")
    grid_at_station = {}
    for _, row in station_locs.iterrows():
        wmo = int(row['ID_WMO'])
        ts = test.sel(lat=float(row['Lat']), lon=float(row['Lon']),
                      method='nearest')
        grid_at_station[wmo] = ts.values
    grid_df = pd.DataFrame(
        grid_at_station,
        index=pd.DatetimeIndex(test.time.values),
    )

    logging.info("[stratified] Aggregating to dekad totals...")
    obs_dk = aggregate_station_df_to_dekad(obs_df)
    grid_dk = aggregate_station_df_to_dekad(grid_df)

    # Attach dekad-of-year to index
    doy_obs = _dekad_of_year_index(obs_dk.index)
    doy_grid = _dekad_of_year_index(grid_dk.index)

    common_stations = sorted(set(obs_dk.columns) & set(grid_dk.columns))
    logging.info(f"[stratified] Computing metrics for {len(common_stations)} stations x 36 doys...")

    rows = []
    for wmo in common_stations:
        obs_ts = obs_dk[wmo]
        grid_ts = grid_dk[wmo]
        for d in range(1, 37):
            mask_o = (doy_obs == d)
            mask_g = (doy_grid == d)
            sub_obs = obs_ts[mask_o].dropna()
            sub_grid = grid_ts[mask_g].dropna()
            common = sub_obs.index.intersection(sub_grid.index)
            if len(common) < min_valid_years:
                continue
            a = sub_obs.loc[common].values.astype(np.float64)
            b = sub_grid.loc[common].values.astype(np.float64)
            metrics = compute_pixel_metrics(a, b, threshold)
            row = {'station_id': wmo, 'dekad_of_year': d,
                   'n_valid_years': len(common)}
            for i, name in enumerate(METRIC_NAMES):
                row[name] = metrics[i]
            rows.append(row)

    return pd.DataFrame(rows).set_index(['station_id', 'dekad_of_year'])
