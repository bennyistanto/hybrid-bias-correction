"""
Module: station_validation.py

Independent validation of bias-corrected precipitation products against
weather station observations.

This module compares gridded bias-corrected products (e.g., LSEQM, LSEQMDL)
against point-scale daily precipitation measurements from BMKG weather
stations. This provides an independent ground-truth validation that is
separate from the QA framework (which compares corrected IMERG vs CPC).

The validation workflow:
1. Load station locations and daily precipitation observations
2. Extract gridded product values at station locations (nearest grid cell)
3. Compute per-station verification metrics using compute_pixel_metrics()
4. Multi-threshold categorical verification (WMO/TD-No. 1485 compliant)
5. Aggregate results into summary statistics

Key design decisions:
- Reuses ``compute_pixel_metrics()`` from metrics.py for consistency
- Reuses ``load_station_locations()`` from station_density.py (DRY)
- Handles BMKG missing data sentinel (8888.0) and empty cells
- Station data date format: DD-MM-YYYY
- Multi-threshold verification follows WMO/WGNE standard contingency table
  metrics: POD, FAR, CSI, FBI, ETS, HSS, HK

References
----------
- WMO (2023), Guidelines for the WMO Evaluation of Records of Weather and
  Climate Extremes, WMO-No. 1317.
- WMO (2009), Recommendations for the Verification and Intercomparison of
  QPFs and PQPFs from Operational NWP Models (Revision 2),
  WMO/TD-No. 1485, WWRP 2009-1.
- WMO (2008), Guide to Meteorological Instruments and Methods of
  Observation, WMO-No. 8.
- WMO (2018), Guide to Climatological Practices, WMO-No. 100.
- Ebert, E. (2007), Methods for verifying satellite precipitation estimates.
- Wilks, D. S. (2011), Statistical Methods in the Atmospheric Sciences,
  3rd ed., Academic Press.

Author
------
Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import logging

from .metrics import compute_pixel_metrics, METRIC_NAMES
from .station_density import load_station_locations

# Missing sentinel and minimum valid days are loaded from config at call time.
# See config.yml → station_validation.missing_sentinel, station_validation.min_valid_days.
def _missing_sentinel():
    """Return BMKG missing-data sentinel from config."""
    from . import config
    return config.MISSING_SENTINEL

def _min_valid_days():
    """Return minimum valid days from config."""
    from . import config
    return config.MIN_VALID_DAYS

# ---------------------------------------------------------------------------
# WMO-compliant multi-threshold verification
# ---------------------------------------------------------------------------
# Thresholds for 24h accumulation (mm/day).
# Sources:
#   WMO/TD-No. 1485 (WWRP 2009-1), Section 3.2: 1, 2, 5, 10, 20, 50 mm
#   BMKG operational intensity classes (WMO Region V):
#     Light 1-5, Moderate 5-20, Heavy 20-50, Very Heavy 50-100, Extreme >100
#
# Combined set covers WMO/TD-1485 thresholds + BMKG intensity boundaries.
# 150 mm included for tropical extreme events (WMO-No. 1317, Table 2).
WMO_THRESHOLDS = (1, 5, 10, 20, 50, 100, 150)

# Labels matching BMKG intensity classification
WMO_THRESHOLD_LABELS = {
    1:   'measurable',    # WMO wet-day threshold (WMO-No. 8)
    5:   'light',         # BMKG light -> moderate boundary
    10:  'moderate_low',  # WMO/TD-1485 threshold
    20:  'moderate',      # BMKG moderate -> heavy boundary
    50:  'heavy',         # BMKG heavy -> very heavy boundary
    100: 'very_heavy',    # BMKG extreme threshold
    150: 'extreme',       # Tropical extreme (WMO-No. 1317)
}

# Minimum number of observed events at a threshold for reliable verification.
# WMO/TD-No. 1485: "verification should not be carried out for thresholds
# where there are fewer than 10 occurrences in the dataset."
MIN_EVENTS_FOR_VERIFICATION = 10

# Multi-threshold metric names (WMO 2x2 contingency table standard)
# Ref: WMO/TD-No. 1485, WGNE Suggested Methods for Precipitation Verification
MULTI_THRESHOLD_METRIC_NAMES = [
    'pod',       # Probability of Detection:   a / (a + c)
    'far',       # False Alarm Ratio:          b / (a + b)
    'csi',       # Critical Success Index:     a / (a + b + c)
    'fbi',       # Frequency Bias Index:       (a + b) / (a + c)
    'ets',       # Equitable Threat Score:     (a - a_r) / (a + b + c - a_r)
    'hss',       # Heidke Skill Score:         2(ad - bc) / [(a+c)(c+d) + (a+b)(b+d)]
    'hk',        # Hanssen-Kuipers Score:      POD - POFD
    'freq_obs',  # Exceedance frequency (obs): n_obs>=thr / N
    'freq_prd',  # Exceedance frequency (prd): n_prd>=thr / N
    'n_events',  # Number of observed events at threshold
]


# +++++++++++++++++++++++++++++++++++++++++
# Data Loading
# +++++++++++++++++++++++++++++++++++++++++

def load_station_observations(station_data_file, station_location_file=None):
    """
    Load daily precipitation observations from a BMKG station data CSV.

    Handles:
    - Date parsing (DD-MM-YYYY format)
    - Missing value replacement (8888.0 -> NaN, empty -> NaN)
    - Optional filtering to only stations present in the location file

    Parameters
    ----------
    station_data_file : str
        Path to CSV with columns: ID, Date, JD, then WMO station ID columns.
    station_location_file : str, optional
        If provided, filters to only stations found in the location CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with DatetimeIndex and WMO station ID columns (as int)
        containing daily precipitation in mm. Shape: (n_days, n_stations).
    """
    logging.info("Loading station observations from %s", station_data_file)

    df = pd.read_csv(station_data_file)

    # Parse dates (DD-MM-YYYY format)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.set_index('Date').sort_index()

    # Drop non-station columns (ID, JD)
    cols_to_drop = [c for c in ['ID', 'JD'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Convert column names to integers (WMO station IDs)
    df.columns = [int(c) for c in df.columns]

    # Replace BMKG missing sentinel (8888.0) with NaN
    df = df.replace(_missing_sentinel(), np.nan)

    # Convert all columns to float (handles any remaining string values)
    df = df.apply(pd.to_numeric, errors='coerce')

    logging.info("Loaded %d days x %d stations", len(df), len(df.columns))
    logging.info("Date range: %s to %s", df.index.min(), df.index.max())

    # Filter to stations present in location file
    if station_location_file is not None:
        loc_df = load_station_locations(station_location_file)
        valid_wmos = set(loc_df['ID_WMO'].astype(int).values)
        available_wmos = set(df.columns)
        common_wmos = sorted(valid_wmos & available_wmos)
        df = df[common_wmos]
        logging.info("Filtered to %d stations present in location file", len(common_wmos))

    # Report data availability
    total_cells = df.size
    valid_cells = df.notna().sum().sum()
    pct_valid = 100.0 * valid_cells / total_cells if total_cells > 0 else 0
    logging.info("Data availability: %.1f%% valid (%d / %d cells)",
                 pct_valid, valid_cells, total_cells)

    return df


# +++++++++++++++++++++++++++++++++++++++++
# Grid Extraction
# +++++++++++++++++++++++++++++++++++++++++

def extract_gridded_at_stations(gridded_da, station_df):
    """
    Extract gridded product timeseries at station locations using nearest
    grid cell.

    Parameters
    ----------
    gridded_da : xarray.DataArray
        Gridded precipitation product with dims (time, lat, lon).
    station_df : pandas.DataFrame
        Station locations with 'Lon', 'Lat', 'ID_WMO' columns
        (from load_station_locations).

    Returns
    -------
    pandas.DataFrame
        DataFrame with DatetimeIndex and WMO station ID columns containing
        gridded product values at nearest grid cell.
    """
    logging.info("Extracting gridded values at %d station locations",
                 len(station_df))

    result = {}
    for _, row in station_df.iterrows():
        wmo_id = int(row['ID_WMO'])
        lat = float(row['Lat'])
        lon = float(row['Lon'])

        # Extract nearest grid cell
        ts = gridded_da.sel(lat=lat, lon=lon, method='nearest')
        result[wmo_id] = ts.values

    # Build DataFrame with matching time index
    times = pd.DatetimeIndex(gridded_da.time.values)
    gridded_df = pd.DataFrame(result, index=times)

    logging.info("Extracted %d timesteps x %d stations", len(gridded_df),
                 len(gridded_df.columns))

    return gridded_df


# +++++++++++++++++++++++++++++++++++++++++
# Per-Station Metrics
# +++++++++++++++++++++++++++++++++++++++++

def compute_station_metrics(obs_df, gridded_df, threshold=1.0):
    """
    Compute per-station validation metrics.

    For each station column present in both DataFrames, extracts the
    overlapping time period, aligns, and calls ``compute_pixel_metrics()``
    from metrics.py.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Station observations (DatetimeIndex, station ID columns).
    gridded_df : pandas.DataFrame
        Gridded values at stations (DatetimeIndex, station ID columns).
    threshold : float, optional
        Wet-day threshold in mm/day (default 1.0).

    Returns
    -------
    pandas.DataFrame
        Per-station metrics table. Rows = stations (WMO IDs),
        Columns = 31 metric names from METRIC_NAMES.
        Includes additional columns: 'n_valid_days', 'station_name',
        'lon', 'lat' (if station metadata is later merged).
    """
    common_stations = sorted(set(obs_df.columns) & set(gridded_df.columns))
    logging.info("Computing metrics for %d common stations", len(common_stations))

    results = []
    skipped = 0

    for wmo_id in common_stations:
        # Get overlapping time period
        obs_ts = obs_df[wmo_id]
        grid_ts = gridded_df[wmo_id]

        # Align to common dates
        common_idx = obs_ts.dropna().index.intersection(grid_ts.dropna().index)

        if len(common_idx) < _min_valid_days():
            skipped += 1
            continue

        ref_1d = obs_ts.loc[common_idx].values.astype(np.float64)
        test_1d = grid_ts.loc[common_idx].values.astype(np.float64)

        # Compute all 31 metrics using existing function
        metrics_tuple = compute_pixel_metrics(ref_1d, test_1d, threshold)

        row = {'station_id': wmo_id, 'n_valid_days': len(common_idx)}
        for i, name in enumerate(METRIC_NAMES):
            row[name] = metrics_tuple[i]

        results.append(row)

    if skipped > 0:
        logging.info("Skipped %d stations with < %d valid paired days",
                     skipped, _min_valid_days())

    if not results:
        logging.warning("No stations had sufficient data for validation")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(results).set_index('station_id')
    logging.info("Computed metrics for %d stations", len(metrics_df))

    return metrics_df


# +++++++++++++++++++++++++++++++++++++++++
# WMO Multi-Threshold Verification
# +++++++++++++++++++++++++++++++++++++++++

def _compute_contingency_table(ref_1d, test_1d, threshold):
    """
    Build the WMO standard 2x2 contingency table for a given threshold.

    The contingency table follows WMO/TD-No. 1485 (WWRP 2009-1) notation::

                      |  Obs >= thr  |  Obs < thr  |
        Prd >= thr    |    a (hits)  |   b (FA)    |
        Prd <  thr    |    c (miss)  |   d (CN)    |

    Parameters
    ----------
    ref_1d : numpy.ndarray
        1-D observed (station) precipitation values.
    test_1d : numpy.ndarray
        1-D predicted (gridded product) precipitation values.
    threshold : float
        Precipitation threshold in mm/day.

    Returns
    -------
    tuple of (float, float, float, float, int)
        (a, b, c, d, N) — hits, false alarms, misses, correct negatives,
        and total number of valid pairs.
    """
    obs_event = ref_1d >= threshold
    prd_event = test_1d >= threshold

    a = float(np.sum(obs_event & prd_event))        # hits
    b = float(np.sum(~obs_event & prd_event))        # false alarms
    c = float(np.sum(obs_event & ~prd_event))        # misses
    d = float(np.sum(~obs_event & ~prd_event))       # correct negatives
    N = int(a + b + c + d)

    return a, b, c, d, N


def _compute_threshold_scores(a, b, c, d, N):
    """
    Compute WMO standard categorical verification scores from a 2x2
    contingency table.

    All formulas follow WMO/TD-No. 1485 (WWRP 2009-1) and the
    WWRP/WGNE Joint Working Group on Forecast Verification Research.

    Parameters
    ----------
    a : float
        Hits.
    b : float
        False alarms.
    c : float
        Misses.
    d : float
        Correct negatives.
    N : int
        Total valid pairs (a + b + c + d).

    Returns
    -------
    dict
        Dictionary with keys: pod, far, csi, fbi, ets, hss, hk,
        freq_obs, freq_prd, n_events.
    """
    # --- POD (Probability of Detection / Hit Rate) ---
    # POD = a / (a + c)           [WMO/TD-1485]
    # Range: 0-1, perfect = 1
    pod = a / (a + c) if (a + c) > 0 else np.nan

    # --- FAR (False Alarm Ratio) ---
    # FAR = b / (a + b)           [WMO/TD-1485]
    # Range: 0-1, perfect = 0
    far = b / (a + b) if (a + b) > 0 else np.nan

    # --- CSI (Critical Success Index / Threat Score) ---
    # CSI = a / (a + b + c)       [WMO/TD-1485]
    # Range: 0-1, perfect = 1
    csi = a / (a + b + c) if (a + b + c) > 0 else np.nan

    # --- FBI (Frequency Bias Index) ---
    # FBI = (a + b) / (a + c)     [WMO/TD-1485]
    # Range: 0-inf, perfect = 1 (>1 overforecasting, <1 underforecasting)
    fbi = (a + b) / (a + c) if (a + c) > 0 else np.nan

    # --- ETS (Equitable Threat Score / Gilbert Skill Score) ---
    # a_random = (a + b)(a + c) / N
    # ETS = (a - a_random) / (a + b + c - a_random)    [WMO/TD-1485]
    # Range: -1/3 to 1, no skill = 0, perfect = 1
    if N > 0:
        a_random = (a + b) * (a + c) / N
        denom = a + b + c - a_random
        ets = (a - a_random) / denom if denom != 0 else np.nan
    else:
        ets = np.nan

    # --- HSS (Heidke Skill Score) ---
    # HSS = 2(ad - bc) / [(a + c)(c + d) + (a + b)(b + d)]   [Wilks 2011]
    # Range: -1 to 1, no skill = 0, perfect = 1
    hss_denom = (a + c) * (c + d) + (a + b) * (b + d)
    hss = 2.0 * (a * d - b * c) / hss_denom if hss_denom > 0 else np.nan

    # --- HK (Hanssen-Kuipers Discriminant / Peirce Skill Score) ---
    # HK = POD - POFD,  where POFD = b / (b + d)   [Wilks 2011]
    # Range: -1 to 1, no skill = 0, perfect = 1
    # Base-rate independent (Hanssen & Kuipers, 1965)
    pofd = b / (b + d) if (b + d) > 0 else np.nan
    if pod is not np.nan and pofd is not np.nan:
        hk = pod - pofd
    else:
        hk = np.nan

    # --- Exceedance frequencies ---
    freq_obs = (a + c) / N if N > 0 else np.nan
    freq_prd = (a + b) / N if N > 0 else np.nan

    return {
        'pod': pod, 'far': far, 'csi': csi, 'fbi': fbi,
        'ets': ets, 'hss': hss, 'hk': hk,
        'freq_obs': freq_obs, 'freq_prd': freq_prd,
        'n_events': int(a + c),
    }


def compute_multi_threshold_metrics(obs_df, gridded_df,
                                    thresholds=None):
    """
    Compute WMO-compliant categorical verification at multiple precipitation
    thresholds for each station.

    For each station and each threshold, builds the 2x2 contingency table
    and derives POD, FAR, CSI, FBI, ETS, HSS, and HK following
    WMO/TD-No. 1485 (WWRP 2009-1).

    Thresholds where a station has fewer than ``MIN_EVENTS_FOR_VERIFICATION``
    observed events are flagged with NaN (per WMO/TD-1485 guidance).

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Station observations (DatetimeIndex, WMO station ID columns).
    gridded_df : pandas.DataFrame
        Gridded values at stations (DatetimeIndex, WMO station ID columns).
    thresholds : sequence of float, optional
        Precipitation thresholds in mm/day. Default: ``WMO_THRESHOLDS``
        (1, 5, 10, 20, 50, 100, 150).

    Returns
    -------
    pandas.DataFrame
        Per-station multi-threshold metrics. Rows = station WMO IDs,
        Columns = ``{metric}_{threshold}mm`` (e.g., ``pod_20mm``,
        ``csi_50mm``, ``ets_100mm``). Also includes ``n_valid_days``.
    """
    if thresholds is None:
        thresholds = WMO_THRESHOLDS

    common_stations = sorted(set(obs_df.columns) & set(gridded_df.columns))
    logging.info("Computing multi-threshold metrics (%d thresholds) for "
                 "%d stations", len(thresholds), len(common_stations))

    results = []
    skipped = 0

    for wmo_id in common_stations:
        obs_ts = obs_df[wmo_id]
        grid_ts = gridded_df[wmo_id]

        # Align to common valid dates
        common_idx = obs_ts.dropna().index.intersection(grid_ts.dropna().index)
        if len(common_idx) < _min_valid_days():
            skipped += 1
            continue

        ref_1d = obs_ts.loc[common_idx].values.astype(np.float64)
        test_1d = grid_ts.loc[common_idx].values.astype(np.float64)

        row = {'station_id': wmo_id, 'n_valid_days': len(common_idx)}

        for thr in thresholds:
            a, b, c, d, N = _compute_contingency_table(ref_1d, test_1d, thr)
            scores = _compute_threshold_scores(a, b, c, d, N)

            # WMO/TD-1485: suppress scores when observed events < 10
            n_obs_events = int(a + c)
            insufficient = n_obs_events < MIN_EVENTS_FOR_VERIFICATION

            suffix = f"_{int(thr)}mm"
            for metric_name in MULTI_THRESHOLD_METRIC_NAMES:
                value = scores[metric_name]
                # Keep n_events and freq_obs even when insufficient
                if insufficient and metric_name not in ('n_events', 'freq_obs',
                                                         'freq_prd'):
                    value = np.nan
                row[metric_name + suffix] = value

        results.append(row)

    if skipped > 0:
        logging.info("Skipped %d stations with < %d valid paired days",
                     skipped, _min_valid_days())

    if not results:
        logging.warning("No stations had sufficient data for multi-threshold "
                        "verification")
        return pd.DataFrame()

    mt_df = pd.DataFrame(results).set_index('station_id')
    logging.info("Multi-threshold metrics computed for %d stations",
                 len(mt_df))

    return mt_df


def summarize_multi_threshold(mt_df, thresholds=None):
    """
    Summarize multi-threshold metrics across all stations.

    Produces a compact table: rows = thresholds, columns = metric medians
    (and IQR). This is the primary output for comparing how skill degrades
    with increasing precipitation intensity.

    Parameters
    ----------
    mt_df : pandas.DataFrame
        Per-station multi-threshold metrics from
        ``compute_multi_threshold_metrics()``.
    thresholds : sequence of float, optional
        Thresholds to include. Default: ``WMO_THRESHOLDS``.

    Returns
    -------
    pandas.DataFrame
        Summary table with index = threshold (mm), columns = metric
        statistics (e.g., ``pod_median``, ``csi_median``, ``ets_p25``).
    """
    if mt_df.empty:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = WMO_THRESHOLDS

    summary_rows = []
    for thr in thresholds:
        suffix = f"_{int(thr)}mm"
        row = {
            'threshold_mm': int(thr),
            'label': WMO_THRESHOLD_LABELS.get(thr, f'{thr}mm'),
        }

        for metric in ('pod', 'far', 'csi', 'fbi', 'ets', 'hss', 'hk',
                        'freq_obs', 'freq_prd'):
            col = metric + suffix
            if col not in mt_df.columns:
                continue
            vals = mt_df[col].dropna()
            if len(vals) == 0:
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_p25'] = np.nan
                row[f'{metric}_p75'] = np.nan
                row[f'{metric}_n_stations'] = 0
            else:
                row[f'{metric}_median'] = vals.median()
                row[f'{metric}_p25'] = vals.quantile(0.25)
                row[f'{metric}_p75'] = vals.quantile(0.75)
                row[f'{metric}_n_stations'] = int(len(vals))

        # Mean observed events
        n_col = 'n_events' + suffix
        if n_col in mt_df.columns:
            row['mean_obs_events'] = mt_df[n_col].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index('threshold_mm')
    return summary_df


# +++++++++++++++++++++++++++++++++++++++++
# Summary & I/O
# +++++++++++++++++++++++++++++++++++++++++

def summarize_station_metrics(metrics_df):
    """
    Aggregate per-station metrics into summary statistics.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-station metrics from ``compute_station_metrics()``.

    Returns
    -------
    pandas.DataFrame
        Summary statistics (count, mean, median, std, min, max) for each
        metric across all stations. Rows = statistic names,
        Columns = metric names.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    # Select only numeric metric columns (exclude n_valid_days for summary)
    metric_cols = [c for c in metrics_df.columns if c in METRIC_NAMES]
    summary = metrics_df[metric_cols].describe().loc[
        ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    ]

    # Rename 50% to median for clarity
    summary = summary.rename(index={'50%': 'median', '25%': 'p25', '75%': 'p75'})

    return summary


def save_station_validation(metrics_df, station_df, output_file,
                            method_name='LSEQMDL'):
    """
    Save per-station validation results to CSV.

    Merges metrics with station metadata (name, lon, lat, elevation) and
    saves as CSV.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-station metrics (index = WMO station IDs).
    station_df : pandas.DataFrame
        Station metadata with ID_WMO, Station, Lon, Lat, Elevation.
    output_file : str
        Output file path (.csv).
    method_name : str, optional
        Name of correction method for metadata.

    Returns
    -------
    str
        Path to saved file.
    """
    if metrics_df.empty:
        logging.warning("No metrics to save (empty DataFrame)")
        return None

    # Merge with station metadata
    station_info = station_df[['ID_WMO', 'Station', 'Lon', 'Lat', 'Elevation']].copy()
    station_info['ID_WMO'] = station_info['ID_WMO'].astype(int)
    station_info = station_info.set_index('ID_WMO')

    merged = metrics_df.join(station_info, how='left')

    # Reorder columns: metadata first, then metrics
    meta_cols = ['Station', 'Lon', 'Lat', 'Elevation', 'n_valid_days']
    metric_cols = [c for c in merged.columns if c not in meta_cols]
    col_order = [c for c in meta_cols if c in merged.columns] + metric_cols
    merged = merged[col_order]

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, float_format='%.6f')
    logging.info("Saved station validation results (%s) to %s",
                 method_name, output_file)

    return output_file


# +++++++++++++++++++++++++++++++++++++++++
# Regional Aggregation
# +++++++++++++++++++++++++++++++++++++++++

# Province-to-region mapping loaded from config at call time.
# See config.yml → region_mapping section.
def _region_mapping():
    """Return province-to-region mapping from config."""
    from . import config
    return config.REGION_MAPPING


def merge_station_metadata(metrics_df, station_df):
    """
    Merge per-station metrics with station location metadata, including
    optional Region and Province columns.

    If the station location CSV contains 'Region' and/or 'Province' columns,
    they are included. If 'Province' exists but 'Region' does not, region is
    inferred from ``config.REGION_MAPPING``.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-station metrics (index = WMO station IDs).
    station_df : pandas.DataFrame
        Station metadata from ``load_station_locations()``.

    Returns
    -------
    pandas.DataFrame
        Metrics with metadata columns prepended: Station, Lon, Lat,
        Elevation, and optionally Region, Province.
    """
    if metrics_df.empty:
        return metrics_df

    # Normalize common column name variants from station CSV
    # (e.g., 'region' -> 'Region', 'a1name' -> 'Province', 'a2name' -> 'District')
    rename_map = {}
    for col in station_df.columns:
        lc = col.lower()
        if lc == 'region' and col != 'Region':
            rename_map[col] = 'Region'
        elif lc in ('a1name', 'province') and col != 'Province':
            rename_map[col] = 'Province'
        elif lc in ('a2name', 'district') and col != 'District':
            rename_map[col] = 'District'
    if rename_map:
        station_df = station_df.rename(columns=rename_map)
        logging.debug("Renamed station columns: %s", rename_map)

    # Determine which metadata columns are available
    meta_cols = ['ID_WMO', 'Station', 'Lon', 'Lat', 'Elevation']
    for optional in ('Region', 'Province', 'District'):
        if optional in station_df.columns:
            meta_cols.append(optional)

    station_info = station_df[meta_cols].copy()
    station_info['ID_WMO'] = station_info['ID_WMO'].astype(int)
    station_info = station_info.set_index('ID_WMO')

    # Drop columns already present in metrics_df to avoid overlap errors
    overlap = set(station_info.columns) & set(metrics_df.columns)
    if overlap:
        station_info = station_info.drop(columns=list(overlap))

    merged = metrics_df.join(station_info, how='left')

    # Infer Region from Province if Province exists but Region does not
    if 'Province' in merged.columns and 'Region' not in merged.columns:
        merged['Region'] = merged['Province'].map(_region_mapping())
        merged['Region'] = merged['Region'].fillna('Other')
        logging.info("Inferred Region from Province using config.REGION_MAPPING")

    return merged


def summarize_by_group(metrics_df, group_col, key_metrics=None):
    """
    Summarize station validation metrics grouped by region or province.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-station metrics with a group column (e.g., 'Region' or
        'Province') already merged via ``merge_station_metadata()``.
    group_col : str
        Column name to group by (e.g., 'Region', 'Province').
    key_metrics : list of str, optional
        Metric column names to summarize. Default: a representative set
        of continuous and categorical metrics.

    Returns
    -------
    pandas.DataFrame
        Summary table: index = group name, columns = metric statistics
        (count, median, mean, std). Groups are sorted alphabetically.
    """
    if metrics_df.empty or group_col not in metrics_df.columns:
        logging.warning("Cannot group by '%s' — column not found", group_col)
        return pd.DataFrame()

    if key_metrics is None:
        key_metrics = [
            'pearson_correlation', 'relative_bias', 'rmse', 'nse',
            'pod', 'far', 'csi', 'ks_pvalue',
        ]

    # Filter to columns that exist
    key_metrics = [m for m in key_metrics if m in metrics_df.columns]

    rows = []
    for group_name, group_df in metrics_df.groupby(group_col):
        row = {group_col: group_name, 'n_stations': len(group_df)}
        for m in key_metrics:
            vals = group_df[m].dropna()
            row[f'{m}_median'] = vals.median() if len(vals) > 0 else np.nan
            row[f'{m}_mean'] = vals.mean() if len(vals) > 0 else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).set_index(group_col).sort_index()
    return summary


# +++++++++++++++++++++++++++++++++++++++++
# Per-Station Scatter Plot
# +++++++++++++++++++++++++++++++++++++++++

def plot_station_scatter(obs_df, gridded_dict, station_id, station_df=None,
                         threshold=1.0, figsize=(12, 5)):
    """
    Generate a scatter plot of observed vs. predicted daily precipitation
    for a single station, with one panel per correction method.

    This is an on-demand function — the user specifies which station to
    plot, avoiding mass generation of figures.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Station observations (DatetimeIndex, WMO station ID columns).
    gridded_dict : dict of {str: pandas.DataFrame}
        Extracted gridded values keyed by method name (e.g.,
        {'LS': df, 'LSEQM': df, 'LSEQMDL': df}).
    station_id : int
        WMO station ID to plot.
    station_df : pandas.DataFrame, optional
        Station metadata for title annotation.
    threshold : float, optional
        Wet-day threshold line (default 1.0 mm).
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure (also displayed via plt.show).
    """
    import matplotlib.pyplot as plt

    methods = list(gridded_dict.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(figsize[0], figsize[1]),
                             squeeze=False)
    axes = axes.flatten()

    # Station name for title
    station_name = f"WMO {station_id}"
    if station_df is not None and 'ID_WMO' in station_df.columns:
        match = station_df[station_df['ID_WMO'].astype(int) == int(station_id)]
        if len(match) > 0:
            row = match.iloc[0]
            station_name = f"{row.get('Station', '')} ({station_id})"
            if 'Province' in row.index and pd.notna(row['Province']):
                station_name += f", {row['Province']}"

    for ax, method_name in zip(axes, methods):
        gridded_df = gridded_dict[method_name]

        if station_id not in obs_df.columns or station_id not in gridded_df.columns:
            ax.set_title(f'{method_name}\n(station not available)')
            continue

        obs_ts = obs_df[station_id]
        grid_ts = gridded_df[station_id]
        common_idx = obs_ts.dropna().index.intersection(grid_ts.dropna().index)

        if len(common_idx) < _min_valid_days():
            ax.set_title(f'{method_name}\n(< {_min_valid_days()} paired days)')
            continue

        obs_vals = obs_ts.loc[common_idx].values
        grid_vals = grid_ts.loc[common_idx].values

        # Compute quick stats for annotation
        from .metrics import compute_pixel_metrics
        metrics_tuple = compute_pixel_metrics(obs_vals, grid_vals, threshold)
        corr = metrics_tuple[1]   # pearson_correlation
        rmse = metrics_tuple[2]   # rmse
        nse = metrics_tuple[14]   # nse
        rb = metrics_tuple[0]     # relative_bias

        # Scatter plot
        ax.scatter(obs_vals, grid_vals, s=8, alpha=0.3, color='steelblue',
                   edgecolors='none')

        # 1:1 line
        max_val = max(obs_vals.max(), grid_vals.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.6,
                label='1:1 line')

        # Threshold lines
        ax.axvline(x=threshold, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(y=threshold, color='grey', linewidth=0.5, linestyle=':')

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel('Observed (mm/day)')
        ax.set_ylabel('Product (mm/day)')
        ax.set_aspect('equal')
        ax.set_title(method_name)
        ax.grid(True, alpha=0.2)

        # Metrics annotation
        stats_text = (f"r = {corr:.3f}\n"
                      f"NSE = {nse:.3f}\n"
                      f"RMSE = {rmse:.1f}\n"
                      f"RB = {rb:+.3f}\n"
                      f"n = {len(common_idx)}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='grey'))

    fig.suptitle(f'Observed vs. Product: {station_name}', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig


# +++++++++++++++++++++++++++++++++++++++++
# Per-Station Daily Precipitation Time Series
# +++++++++++++++++++++++++++++++++++++++++

def plot_station_timeseries(obs_df, gridded_dict, station_id, station_df=None,
                            thresholds=(5, 10, 20, 50, 100),
                            figsize=(16, 6)):
    """
    Generate a daily precipitation dot plot for a single station over time.

    Each dataset (observed + gridded products) is plotted as semi-transparent
    dots along the time axis. Horizontal dashed lines mark WMO rainfall
    intensity thresholds for visual context.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Station observations (DatetimeIndex, WMO station ID columns).
    gridded_dict : dict of {str: pandas.DataFrame}
        Extracted gridded values keyed by method name (e.g.,
        {'LS': df, 'LSEQM': df, 'LSEQMDL': df}).
    station_id : int
        WMO station ID to plot.
    station_df : pandas.DataFrame, optional
        Station metadata for title annotation.
    thresholds : tuple of float, optional
        Precipitation thresholds (mm/day) for horizontal reference lines.
        Default: (5, 10, 20, 50, 100).
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure (also displayed via plt.show).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Color palette for datasets
    dataset_colors = {
        'BMKG': '#222222',
        'IMERG': '#999999',
        'CPC': '#666666',
        'LS': '#e74c3c',
        'LSEQM': '#f39c12',
        'LSEQMDL': '#27ae60',
    }

    # Station name for title
    station_name = f"WMO {station_id}"
    province_str = ""
    if station_df is not None and 'ID_WMO' in station_df.columns:
        match = station_df[station_df['ID_WMO'].astype(int) == int(station_id)]
        if len(match) > 0:
            row = match.iloc[0]
            station_name = f"{row.get('Station', '')} ({station_id})"
            # Check for Province / a1name
            for pcol in ('Province', 'a1name'):
                if pcol in row.index and pd.notna(row[pcol]):
                    province_str = str(row[pcol])
                    break

    fig, ax = plt.subplots(figsize=figsize)

    # --- Plot observed data ---
    if station_id in obs_df.columns:
        obs_ts = obs_df[station_id].dropna()
        if len(obs_ts) > 0:
            ax.scatter(obs_ts.index, obs_ts.values, s=12, alpha=0.35,
                       color=dataset_colors.get('BMKG', '#222222'),
                       edgecolors='none', label='BMKG Observed', zorder=3)

    # --- Plot gridded products ---
    stats_lines = []
    for method_name, gridded_df in gridded_dict.items():
        if station_id not in gridded_df.columns:
            continue

        grid_ts = gridded_df[station_id].dropna()
        if len(grid_ts) == 0:
            continue

        color = dataset_colors.get(method_name, '#333333')
        ax.scatter(grid_ts.index, grid_ts.values, s=10, alpha=0.3,
                   color=color, edgecolors='none', label=method_name, zorder=2)

        # Compute quick stats against observations
        if station_id in obs_df.columns:
            obs_ts = obs_df[station_id]
            common_idx = obs_ts.dropna().index.intersection(grid_ts.index)
            if len(common_idx) >= _min_valid_days():
                obs_vals = obs_ts.loc[common_idx].values.astype(np.float64)
                grid_vals = grid_ts.loc[common_idx].values.astype(np.float64)
                from .metrics import compute_pixel_metrics
                mt = compute_pixel_metrics(obs_vals, grid_vals, 1.0)
                corr = mt[1]    # pearson_correlation
                rmse = mt[2]    # rmse
                nse = mt[14]    # nse
                stats_lines.append(
                    f"{method_name}: r={corr:.2f}  RMSE={rmse:.1f}  "
                    f"NSE={nse:.2f}  (n={len(common_idx)})"
                )

    # --- WMO threshold reference lines ---
    threshold_colors = {
        5: '#4fc3f7', 10: '#29b6f6', 20: '#0288d1',
        50: '#f57c00', 100: '#d32f2f',
    }
    for thr in thresholds:
        color = threshold_colors.get(thr, '#888888')
        ax.axhline(y=thr, color=color, linestyle='--', linewidth=0.8,
                   alpha=0.6, zorder=1)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.01, thr,
                f' {thr} mm', color=color, fontsize=7,
                va='center', ha='left',
                transform=ax.get_yaxis_transform())

    # --- Formatting ---
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Precipitation (mm/day)')
    title = f'Daily Precipitation: {station_name}'
    if province_str:
        title += f', {province_str}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    # Date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=30)

    # Legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8,
              markerscale=2, scatterpoints=1)

    # Stats annotation box
    if stats_lines:
        stats_text = '\n'.join(stats_lines)
        ax.text(0.01, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.85, edgecolor='grey'))

    plt.tight_layout()
    plt.show()

    return fig


# +++++++++++++++++++++++++++++++++++++++++
# Gridded Metrics / QA Extraction at Stations
# +++++++++++++++++++++++++++++++++++++++++

def extract_metrics_at_stations(metrics_file, station_df, metric_names=None):
    """
    Extract pixel-level gridded metrics at station locations.

    Opens a metrics NetCDF file (output of ``run_metrics_pipeline``),
    extracts variable values at each station's nearest grid cell, and
    returns a DataFrame.

    Parameters
    ----------
    metrics_file : str
        Path to a metrics NetCDF file (2-D variables: lat × lon).
    station_df : pandas.DataFrame
        Station locations with 'ID_WMO', 'Lon', 'Lat' columns.
    metric_names : list of str, optional
        Variables to extract. If None, extracts all data variables.

    Returns
    -------
    pandas.DataFrame
        Index = WMO station IDs, columns = metric names.
    """
    from .config import NETCDF_ENGINE

    if not os.path.isfile(metrics_file):
        logging.warning("Metrics file not found: %s", metrics_file)
        return pd.DataFrame()

    ds = xr.open_dataset(metrics_file, engine=NETCDF_ENGINE)

    if metric_names is None:
        metric_names = list(ds.data_vars)

    results = {}
    for _, row in station_df.iterrows():
        wmo_id = int(row['ID_WMO'])
        lat = float(row['Lat'])
        lon = float(row['Lon'])

        row_dict = {}
        for var in metric_names:
            if var not in ds:
                continue
            val = ds[var].sel(lat=lat, lon=lon, method='nearest')
            row_dict[var] = float(val.values) if val.size == 1 else np.nan

        results[wmo_id] = row_dict

    ds.close()

    result_df = pd.DataFrame.from_dict(results, orient='index')
    result_df.index.name = 'station_id'
    logging.info("Extracted %d metrics at %d stations from %s",
                 len(metric_names), len(result_df), metrics_file)

    return result_df


def extract_qa_at_stations(qa_file, station_df, var_names=None):
    """
    Extract pixel-level gridded QA values at station locations.

    Opens a QA NetCDF file (output of ``run_qa_pipeline``), extracts
    variable values at each station's nearest grid cell. Handles both
    numeric (score) and categorical (category) variables.

    Parameters
    ----------
    qa_file : str
        Path to a QA NetCDF file (2-D variables: lat × lon).
    station_df : pandas.DataFrame
        Station locations with 'ID_WMO', 'Lon', 'Lat' columns.
    var_names : list of str, optional
        Variables to extract. If None, extracts all data variables.

    Returns
    -------
    pandas.DataFrame
        Index = WMO station IDs, columns = QA variable names.
    """
    from .config import NETCDF_ENGINE

    if not os.path.isfile(qa_file):
        logging.warning("QA file not found: %s", qa_file)
        return pd.DataFrame()

    ds = xr.open_dataset(qa_file, engine=NETCDF_ENGINE)

    if var_names is None:
        var_names = list(ds.data_vars)

    results = {}
    for _, row in station_df.iterrows():
        wmo_id = int(row['ID_WMO'])
        lat = float(row['Lat'])
        lon = float(row['Lon'])

        row_dict = {}
        for var in var_names:
            if var not in ds:
                continue
            val = ds[var].sel(lat=lat, lon=lon, method='nearest')
            raw = val.values
            if raw.size == 1:
                raw = raw.item()
                # Keep as-is (may be int category or float score)
                row_dict[var] = raw
            elif raw.size > 1:
                # qualityts files have a time dimension (per-year);
                # average across years for a representative value
                finite = raw[np.isfinite(raw)]
                row_dict[var] = (
                    float(np.nanmean(finite)) if len(finite) > 0
                    else np.nan
                )
            else:
                row_dict[var] = np.nan

        results[wmo_id] = row_dict

    ds.close()

    result_df = pd.DataFrame.from_dict(results, orient='index')
    result_df.index.name = 'station_id'
    logging.info("Extracted %d QA variables at %d stations from %s",
                 len(var_names), len(result_df), qa_file)

    return result_df


# +++++++++++++++++++++++++++++++++++++++++
# End-to-End Pipeline
# +++++++++++++++++++++++++++++++++++++++++

def run_station_validation(gridded_da, station_location_file, station_data_file,
                           threshold=1.0):
    """
    End-to-end station validation pipeline.

    Chains: load locations -> load observations -> extract gridded at
    stations -> compute metrics -> summarize.

    Parameters
    ----------
    gridded_da : xarray.DataArray
        Bias-corrected gridded precipitation product with dims
        (time, lat, lon).
    station_location_file : str
        Path to station location CSV.
    station_data_file : str
        Path to station observation data CSV.
    threshold : float, optional
        Wet-day threshold in mm/day (default 1.0).

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        (per_station_metrics, summary_stats, station_locations)
    """
    logging.info("=" * 60)
    logging.info("Starting independent station validation")
    logging.info("=" * 60)

    # Load station locations (reusing station_density function)
    station_df = load_station_locations(station_location_file)
    logging.info("Loaded %d station locations", len(station_df))

    # Load observations
    obs_df = load_station_observations(
        station_data_file, station_location_file
    )

    # Extract gridded values at station locations
    gridded_df = extract_gridded_at_stations(gridded_da, station_df)

    # Compute per-station metrics
    metrics_df = compute_station_metrics(obs_df, gridded_df, threshold)

    # Summarize
    summary_df = summarize_station_metrics(metrics_df)

    logging.info("Station validation complete: %d stations evaluated",
                 len(metrics_df))

    return metrics_df, summary_df, station_df
