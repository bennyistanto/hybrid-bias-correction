"""
Module: metrics.py

Performance metrics computation for bias-corrected precipitation products.

This module provides functions for computing pixel-level verification metrics
that compare a test precipitation dataset (e.g., bias-corrected IMERG) against
a reference dataset (e.g., CPC-UNI). Metrics are computed per grid cell over
a specified dekad (10-day period) across multiple years.

Metrics computed (31 total):
  - Continuous:  Relative Bias, Pearson Correlation, RMSE, MAE, NSE,
                 Std Dev (ref/test), Std Dev Ratio, KS-test (stat + p-value)
  - Categorical: POD, FAR, CSI, Frequency of Precipitation Days (ref/test),
                 Mean Wet-Day Precipitation (ref/test), Dry Spell Length (ref/test)
  - Percentiles: p25, p50, p75, p90, p95, p99 (ref and test)

Output is stored as CF-1.8 compliant NetCDF with dimensions (time, lat, lon)
where time has one entry per year (timeseries mode) or (lat, lon) for
single-dekad aggregated mode.

References
----------
- WMO (2008), Guide to Meteorological Instruments and Methods of Observation.
- Wilks, D.S. (2011), Statistical Methods in the Atmospheric Sciences, 3rd ed.
- Ebert, E. (2007), Methods for verifying satellite precipitation estimates.

Author
------
Benny Istanto
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""
import os
import numpy as np
import xarray as xr
import pandas as pd
import logging
from scipy import stats
from scipy.stats import pearsonr

from .config import (
    mask_file, output_filename_template,
    IMERG_PRECIP_VAR, CPC_PRECIP_VAR,
    NETCDF_ENGINE,
)
from .utility import apply_land_sea_mask, set_user_decision
from .utility import ensure_strict_monotonic_time

# +++++++++++++++++++++++++++++++++++++++++
# Constants: CF-1.8 NetCDF encoding for metrics
# +++++++++++++++++++++++++++++++++++++++++

# Variable names and their CF metadata
METRIC_LONG_NAMES = {
    'relative_bias': 'Relative Bias between test and reference',
    'pearson_correlation': 'Pearson Correlation Coefficient',
    'rmse': 'Root Mean Square Error',
    'mae': 'Mean Absolute Error',
    'ref_stdev': 'Standard Deviation of Reference Data',
    'test_stdev': 'Standard Deviation of Test Data',
    'stdev_ratio': 'Ratio of Test to Reference Standard Deviation',
    'pod': 'Probability of Detection',
    'far': 'False Alarm Ratio',
    'csi': 'Critical Success Index',
    'fpd_ref': 'Frequency of Precipitation Days (Reference)',
    'fpd_test': 'Frequency of Precipitation Days (Test)',
    'mdwp_ref': 'Mean Wet-Day Precipitation (Reference)',
    'mdwp_test': 'Mean Wet-Day Precipitation (Test)',
    'nse': 'Nash-Sutcliffe Efficiency',
    'dsl_ref': 'Maximum Dry Spell Length (Reference)',
    'dsl_test': 'Maximum Dry Spell Length (Test)',
    'ks_stat': 'Kolmogorov-Smirnov Test Statistic',
    'ks_pvalue': 'Kolmogorov-Smirnov Test P-Value',
    'p25_ref': '25th Percentile of Reference Data',
    'p25_test': '25th Percentile of Test Data',
    'p50_ref': '50th Percentile (Median) of Reference Data',
    'p50_test': '50th Percentile (Median) of Test Data',
    'p75_ref': '75th Percentile of Reference Data',
    'p75_test': '75th Percentile of Test Data',
    'p90_ref': '90th Percentile of Reference Data',
    'p90_test': '90th Percentile of Test Data',
    'p95_ref': '95th Percentile of Reference Data',
    'p95_test': '95th Percentile of Test Data',
    'p99_ref': '99th Percentile of Reference Data',
    'p99_test': '99th Percentile of Test Data',
}

METRIC_UNITS = {
    'relative_bias': 'ratio', 'pearson_correlation': 'unitless',
    'nse': 'unitless', 'stdev_ratio': 'ratio',
    'pod': 'unitless', 'far': 'unitless', 'csi': 'unitless',
    'fpd_ref': 'percent', 'fpd_test': 'percent',
    'ks_stat': 'unitless', 'ks_pvalue': 'unitless',
    'rmse': 'mm/day', 'mae': 'mm/day',
    'ref_stdev': 'mm/day', 'test_stdev': 'mm/day',
    'mdwp_ref': 'mm/day', 'mdwp_test': 'mm/day',
    'p25_ref': 'mm/day', 'p25_test': 'mm/day',
    'p50_ref': 'mm/day', 'p50_test': 'mm/day',
    'p75_ref': 'mm/day', 'p75_test': 'mm/day',
    'p90_ref': 'mm/day', 'p90_test': 'mm/day',
    'p95_ref': 'mm/day', 'p95_test': 'mm/day',
    'p99_ref': 'mm/day', 'p99_test': 'mm/day',
    'dsl_ref': 'days', 'dsl_test': 'days',
}

# Standard CF-1.8 encoding for all 31 metric variables
CF18_METRICS = {
    name: {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan}
    for name in METRIC_LONG_NAMES
}

# Ordered list of all 31 metric variable names
METRIC_NAMES = list(METRIC_LONG_NAMES.keys())


# +++++++++++++++++++++++++++++++++++++++++
# Core Metric Functions
# +++++++++++++++++++++++++++++++++++++++++

def _compute_max_dry_spell(data, threshold):
    """
    Compute maximum consecutive dry spell length.

    Parameters
    ----------
    data : numpy.ndarray
        1-D array of daily precipitation values.
    threshold : float
        Wet-day threshold in mm/day.

    Returns
    -------
    int
        Length (days) of the longest consecutive run below threshold.
    """
    current_spell = 0
    max_spell = 0
    for val in data:
        if np.isnan(val) or val < threshold:
            current_spell += 1
            max_spell = max(max_spell, current_spell)
        else:
            current_spell = 0
    return max_spell


def compute_pixel_metrics(ref_1d, test_1d, threshold=1.0):
    """
    Compute all 31 verification metrics for a single pixel time series.

    This function is designed to be called via ``xr.apply_ufunc`` with
    ``vectorize=True``, operating on the time dimension of each grid cell.

    Parameters
    ----------
    ref_1d : numpy.ndarray
        1-D reference precipitation values (e.g., CPC) over the dekad window.
    test_1d : numpy.ndarray
        1-D test precipitation values (e.g., bias-corrected IMERG) over the
        same dekad window. Must have the same length as ``ref_1d``.
    threshold : float, optional
        Wet-day threshold in mm/day for categorical metrics (POD, FAR, CSI,
        FPD, MDWP, DSL). Default is 1.0 mm/day.

    Returns
    -------
    tuple of float
        Tuple of 31 metric values in the order defined by ``METRIC_NAMES``.
        Returns all-NaN tuple if insufficient valid data.
    """
    nan_result = tuple([np.nan] * 31)

    # Mask invalid (NaN) entries in both arrays
    valid = ~np.isnan(ref_1d) & ~np.isnan(test_1d)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return nan_result

    ref_clean = ref_1d[valid].astype(np.float64)
    test_clean = test_1d[valid].astype(np.float64)

    # --- Continuous metrics ---

    # Pearson correlation
    if np.std(ref_clean) == 0 or np.std(test_clean) == 0:
        corr = np.nan
    else:
        corr, _ = pearsonr(ref_clean, test_clean)

    # Relative bias: (sum_test - sum_ref) / sum_ref
    sum_ref = np.sum(ref_clean)
    sum_test = np.sum(test_clean)
    rb = (sum_test - sum_ref) / sum_ref if sum_ref != 0 else np.nan
    if not np.isfinite(rb):
        rb = np.nan

    # RMSE
    rmse = np.sqrt(np.mean((test_clean - ref_clean) ** 2))

    # MAE
    mae = np.mean(np.abs(test_clean - ref_clean))

    # Standard deviations (sample, ddof=1)
    ref_stdev = np.std(ref_clean, ddof=1)
    test_stdev = np.std(test_clean, ddof=1)
    stdev_ratio = test_stdev / ref_stdev if ref_stdev != 0 else np.nan
    if not np.isfinite(stdev_ratio):
        stdev_ratio = np.nan

    # Nash-Sutcliffe Efficiency: 1 - SS_res / SS_tot
    ss_tot = np.sum((ref_clean - np.mean(ref_clean)) ** 2)
    if ss_tot > 0:
        nse = 1 - np.sum((ref_clean - test_clean) ** 2) / ss_tot
        if not np.isfinite(nse):
            nse = np.nan
    else:
        nse = np.nan

    # Kolmogorov-Smirnov two-sample test
    ks_stat, ks_pvalue = stats.ks_2samp(ref_clean, test_clean, method='asymp')

    # --- Categorical metrics ---

    hits = float(np.sum((ref_clean >= threshold) & (test_clean >= threshold)))
    misses = float(np.sum((ref_clean >= threshold) & (test_clean < threshold)))
    false_alarms = float(np.sum((ref_clean < threshold) & (test_clean >= threshold)))

    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

    # Frequency of precipitation days (percent of valid days)
    fpd_ref = float(np.sum(ref_clean >= threshold)) / n_valid * 100
    fpd_test = float(np.sum(test_clean >= threshold)) / n_valid * 100

    # Mean wet-day precipitation
    wet_ref = ref_clean[ref_clean >= threshold]
    wet_test = test_clean[test_clean >= threshold]
    mdwp_ref = np.mean(wet_ref) if len(wet_ref) > 0 else np.nan
    mdwp_test = np.mean(wet_test) if len(wet_test) > 0 else np.nan

    # Maximum dry spell length
    dsl_ref = float(_compute_max_dry_spell(ref_clean, threshold))
    dsl_test = float(_compute_max_dry_spell(test_clean, threshold))

    # --- Percentiles ---
    p25_ref = np.nanpercentile(ref_clean, 25)
    p25_test = np.nanpercentile(test_clean, 25)
    p50_ref = np.nanpercentile(ref_clean, 50)
    p50_test = np.nanpercentile(test_clean, 50)
    p75_ref = np.nanpercentile(ref_clean, 75)
    p75_test = np.nanpercentile(test_clean, 75)
    p90_ref = np.nanpercentile(ref_clean, 90)
    p90_test = np.nanpercentile(test_clean, 90)
    p95_ref = np.nanpercentile(ref_clean, 95)
    p95_test = np.nanpercentile(test_clean, 95)
    p99_ref = np.nanpercentile(ref_clean, 99)
    p99_test = np.nanpercentile(test_clean, 99)

    return (
        rb, corr, rmse, mae,
        ref_stdev, test_stdev, stdev_ratio,
        pod, far, csi,
        fpd_ref, fpd_test, mdwp_ref, mdwp_test,
        nse, dsl_ref, dsl_test,
        ks_stat, ks_pvalue,
        p25_ref, p25_test, p50_ref, p50_test,
        p75_ref, p75_test, p90_ref, p90_test,
        p95_ref, p95_test, p99_ref, p99_test,
    )


# +++++++++++++++++++++++++++++++++++++++++
# Dataset-Level Functions
# +++++++++++++++++++++++++++++++++++++++++

def _apply_pixel_metrics(ref_sub, test_sub, threshold=1.0):
    """
    Apply compute_pixel_metrics to two aligned DataArrays using xr.apply_ufunc.

    Parameters
    ----------
    ref_sub : xarray.DataArray
        Reference data for one year's dekad, dims=(time, lat, lon).
    test_sub : xarray.DataArray
        Test data for the same period, dims=(time, lat, lon).
    threshold : float
        Wet-day threshold in mm/day.

    Returns
    -------
    xarray.Dataset
        Dataset with 31 metric variables, dims=(lat, lon).
    """
    def _pixel_func(r, t):
        return compute_pixel_metrics(r, t, threshold)

    result = xr.apply_ufunc(
        _pixel_func,
        ref_sub, test_sub,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[] for _ in range(31)],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float] * 31,
    )

    ds = xr.Dataset({
        name: result[i]
        for i, name in enumerate(METRIC_NAMES)
    })
    return ds


def compute_dekad_metrics_timeseries(ref_da, test_da, threshold=1.0):
    """
    Compute verification metrics per year for a single dekad.

    For each year present in both ``ref_da`` and ``test_da``, the function
    computes 31 pixel-level metrics across the ~10-day dekad window. The
    result has one time entry per year.

    Parameters
    ----------
    ref_da : xarray.DataArray
        Daily reference precipitation, already sliced to the target month/dekad.
        Dims: (time, lat, lon).
    test_da : xarray.DataArray
        Daily test precipitation for the same period. Dims: (time, lat, lon).
    threshold : float, optional
        Wet-day threshold in mm/day. Default is 1.0.

    Returns
    -------
    xarray.Dataset
        Dataset with 31 metric variables, dims=(time, lat, lon) where
        time contains one entry per year.
    """
    # Align in time and space
    ref_da, test_da = xr.align(ref_da, test_da, join='inner')

    # Group by year
    year_groups_ref = ref_da.groupby('time.year')
    year_groups_test = test_da.groupby('time.year')
    common_years = sorted(
        set(year_groups_ref.groups.keys()) & set(year_groups_test.groups.keys())
    )

    n_years = len(common_years)
    logging.info(f"  Computing metrics for {n_years} years...")

    out_list = []
    for idx, yr in enumerate(common_years, 1):
        ref_sub = ref_da.where(ref_da['time.year'] == yr, drop=True)
        test_sub = test_da.where(test_da['time.year'] == yr, drop=True)

        if len(ref_sub.time) == 0:
            continue

        ds_year = _apply_pixel_metrics(ref_sub, test_sub, threshold)
        ds_year = ds_year.expand_dims(dim={'time': [yr]})
        out_list.append(ds_year)

        if idx % 5 == 0 or idx == n_years:
            logging.info(f"    Year {yr} done ({idx}/{n_years})")

    if out_list:
        metrics_ds = xr.concat(out_list, dim='time')
        metrics_ds = metrics_ds.sortby('time')
    else:
        metrics_ds = xr.Dataset()

    return metrics_ds


def compute_single_dekad_metrics(ref_da, test_da, threshold=1.0):
    """
    Compute aggregated verification metrics across all years for a single dekad.

    Unlike ``compute_dekad_metrics_timeseries``, this function pools all
    available time steps together and produces a single (lat, lon) result.

    Parameters
    ----------
    ref_da : xarray.DataArray
        Daily reference precipitation, already sliced to the target month/dekad.
        Dims: (time, lat, lon).
    test_da : xarray.DataArray
        Daily test precipitation for the same period. Dims: (time, lat, lon).
    threshold : float, optional
        Wet-day threshold in mm/day. Default is 1.0.

    Returns
    -------
    xarray.Dataset
        Dataset with 31 metric variables, dims=(lat, lon).
    """
    ref_da, test_da = xr.align(ref_da, test_da, join='inner')
    logging.info(f"  Computing aggregated metrics across {len(ref_da.time)} time steps...")
    result = _apply_pixel_metrics(ref_da, test_da, threshold)
    logging.info(f"  Aggregated metrics complete.")
    return result


# +++++++++++++++++++++++++++++++++++++++++
# I/O Functions
# +++++++++++++++++++++++++++++++++++++++++

def save_metrics(metrics_ds, out_file, description="Bias Correction Metrics"):
    """
    Save a metrics dataset to CF-1.8 compliant NetCDF.

    Parameters
    ----------
    metrics_ds : xarray.Dataset
        Dataset with metric variables to save.
    out_file : str
        Output file path.
    description : str, optional
        Title for the dataset global attribute.

    Returns
    -------
    str or None
        Path to the saved file, or None if skipped.
    """
    if os.path.exists(out_file):
        logging.info(f"File {out_file} already exists.")
        decision = set_user_decision()
        if decision == 'S':
            logging.info(f"Skipping file {out_file}")
            return None
        elif decision == 'A':
            logging.info("Aborting process.")
            from .io import BiasCorrectAbort
            raise BiasCorrectAbort("User chose to abort.")

    # Global metadata
    metrics_ds.attrs.update({
        'cdm_data_type': 'GRID',
        'Conventions': 'CF-1.8',
        'title': description,
        'history': f'Created on {pd.Timestamp.now()}',
        'creator_name': 'Benny Istanto',
        'creator_role': 'Climate Geographer',
        'creator_email': 'bennyistanto@apps.ipb.ac.id',
    })

    # Per-variable metadata
    for var in metrics_ds.data_vars:
        if var in METRIC_LONG_NAMES:
            metrics_ds[var].attrs['long_name'] = METRIC_LONG_NAMES[var]
        if var in METRIC_UNITS:
            metrics_ds[var].attrs['units'] = METRIC_UNITS[var]

    # Coordinate metadata
    if 'lat' in metrics_ds.coords:
        metrics_ds['lat'].attrs.update({
            'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'
        })
    if 'lon' in metrics_ds.coords:
        metrics_ds['lon'].attrs.update({
            'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'
        })

    # Build encoding (only for variables present in the dataset)
    encoding = {var: CF18_METRICS[var] for var in metrics_ds.data_vars if var in CF18_METRICS}

    # Time coordinate encoding: in timeseries mode the time dim holds integer
    # years (e.g. 2001, 2002, ...) not datetime objects.  Encode as int32 to
    # avoid the "overflow encountered in cast" warning that occurs when xarray
    # auto-encodes plain integers through its datetime path.
    if 'time' in metrics_ds.coords:
        encoding['time'] = {'dtype': 'int32', 'zlib': True, '_FillValue': None}

    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        metrics_ds.to_netcdf(out_file, engine=NETCDF_ENGINE, encoding=encoding)
        logging.info(f"Saved metrics to {out_file}")
        return out_file
    except IOError as e:
        logging.error(f"Failed to save {out_file}: {e}")
        return None


def slice_month_dekad(ds, month, dekad):
    """
    Slice a dataset to extract data for a specified month and dekad.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input data with a 'time' dimension.
    month : int
        Month number (1-12).
    dekad : int
        Dekad number (1, 2, or 3).
        - Dekad 1: days 1-10
        - Dekad 2: days 11-20
        - Dekad 3: days 21-end of month

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Subset containing only the matching time steps.
    """
    from .io import get_max_day_in_month

    if dekad == 1:
        start_day, end_day = 1, 10
    elif dekad == 2:
        start_day, end_day = 11, 20
    elif dekad == 3:
        start_day = 21
        end_day = get_max_day_in_month(ds, month)
    else:
        raise ValueError(f"Invalid dekad: {dekad}. Must be 1, 2, or 3.")

    mask = (
        (ds['time.month'] == month) &
        (ds['time.day'] >= start_day) &
        (ds['time.day'] <= end_day)
    )
    return ds.where(mask, drop=True)


# +++++++++++++++++++++++++++++++++++++++++
# CPC Regridding for Metrics
# +++++++++++++++++++++++++++++++++++++++++

def unify_cpc_for_metrics(cpc_ds, test_ds):
    """
    Regrid CPC (0.5 deg) to match a test dataset (0.1 deg) in time and space.

    Steps:
    1. Ensure strictly monotonic time on both datasets.
    2. Select only overlapping time steps (inner join).
    3. Nearest-neighbor interpolation of CPC to test lat/lon grid.

    Parameters
    ----------
    cpc_ds : xarray.Dataset
        CPC-UNI dataset with 'time', 'lat', 'lon' dimensions.
    test_ds : xarray.Dataset
        Test (corrected) dataset to align with.

    Returns
    -------
    xarray.Dataset
        CPC dataset regridded to the test grid with matching time steps.
        Returns empty Dataset if no overlapping times found.
    """
    cpc_ds = ensure_strict_monotonic_time(cpc_ds)
    test_ds = ensure_strict_monotonic_time(test_ds)

    # Find common time steps
    common_times = sorted(
        set(cpc_ds.time.values) & set(test_ds.time.values)
    )
    if len(common_times) == 0:
        logging.warning("unify_cpc_for_metrics: no overlapping time steps.")
        return xr.Dataset()

    common_times = pd.DatetimeIndex(common_times)
    cpc_aligned = cpc_ds.sel(time=common_times)

    # Nearest-neighbor interpolation to test grid
    cpc_regridded = cpc_aligned.interp(
        lat=test_ds.lat, lon=test_ds.lon, method='nearest'
    )

    logging.info(
        f"CPC regridded: {dict(cpc_ds.sizes)} -> {dict(cpc_regridded.sizes)}, "
        f"{len(common_times)} common time steps"
    )

    return cpc_regridded


# +++++++++++++++++++++++++++++++++++++++++
# Metric Combo Builder
# +++++++++++++++++++++++++++++++++++++++++

def build_metric_combos(month_str, dekad_str):
    """
    Build the list of 9 reference-vs-test combinations for metric computation.

    Combinations: {CPC, IMERGL, IMERGF} x {LS, LSEQM, LSEQMDL}.

    Parameters
    ----------
    month_str : str
        Two-digit month string, e.g. '01'.
    dekad_str : str
        Dekad start-day string: '01', '11', or '21'.

    Returns
    -------
    list of dict
        Each dict has keys: ref_label, test_label, test_file, output_dir.
    """
    from . import config

    correction_methods = [
        ('ls', config.ls_corrected_precip_path,
         config.metrics_path_template.replace('{method}', 'ls')),
        ('lseqm', config.lseqm_corrected_precip_path,
         config.metrics_path_template.replace('{method}', 'lseqm')),
        ('lseqmdl', config.lseqmdl_corrected_precip_path,
         config.metrics_path_template.replace('{method}', 'lseqmdl')),
    ]

    ref_labels = ['cpc', 'imergl', 'imergf']

    combos = []
    for ref_label in ref_labels:
        for method_abbr, corrected_path, metrics_dir in correction_methods:
            test_label = f"imergl_{method_abbr}"
            test_file = os.path.join(
                corrected_path,
                f"{config.FILENAME_PREFIX}_{method_abbr}_corrected_imergl_month{month_str}_dekad{dekad_str}.nc4"
            )
            combos.append({
                'ref_label': ref_label,
                'test_label': test_label,
                'test_file': test_file,
                'output_dir': metrics_dir,
            })

    return combos


# +++++++++++++++++++++++++++++++++++++++++
# Main Orchestration Pipeline
# +++++++++++++++++++++++++++++++++++++++++

def run_metrics_pipeline(month, dekad, mode='timeseries'):
    """
    Run the full metrics computation pipeline for one month/dekad.

    Loads CPC, IMERGL, and IMERGF datasets once, then iterates over all
    9 reference-vs-test combinations. For each combo, it:

    1. Regrids CPC to the test grid (if reference is CPC).
    2. Slices data to the target month/dekad window.
    3. Applies the land-sea mask.
    4. Computes 31 verification metrics.
    5. Saves results to CF-1.8 NetCDF.

    Parameters
    ----------
    month : int
        Month number (1-12).
    dekad : int
        Dekad number (1, 2, or 3).
    mode : str, optional
        'timeseries' (default) - per-year metrics, dims=(time, lat, lon).
        'single' - aggregated across all years, dims=(lat, lon).

    Returns
    -------
    list of str
        Paths to the saved NetCDF files (None entries for skipped combos).
    """
    from . import config
    from .io import get_max_day_in_month

    if mode not in ('timeseries', 'single'):
        raise ValueError(f"mode must be 'timeseries' or 'single', got '{mode}'")

    # Format strings for filenames
    month_str = f"{month:02d}"
    if dekad == 1:
        dekad_str = "01"
    elif dekad == 2:
        dekad_str = "11"
    else:
        dekad_str = "21"

    prefix = 'metricsts' if mode == 'timeseries' else 'metricssd'

    # --- Load reference datasets once ---
    logging.info("Loading CPC, IMERGL, IMERGF datasets...")

    ref_datasets = {}

    # CPC
    cpc_ds = xr.open_dataset(config.cpc_file, decode_times=True, engine=NETCDF_ENGINE)
    if cpc_ds.lat[0] > cpc_ds.lat[-1]:
        cpc_ds = cpc_ds.reindex(lat=cpc_ds.lat[::-1])
    ref_datasets['cpc'] = cpc_ds

    # IMERGL
    imergl_ds = xr.open_dataset(config.imergl_file, decode_times=True, engine=NETCDF_ENGINE)
    if imergl_ds.lat[0] > imergl_ds.lat[-1]:
        imergl_ds = imergl_ds.reindex(lat=imergl_ds.lat[::-1])
    ref_datasets['imergl'] = imergl_ds

    # IMERGF
    if os.path.isfile(config.imergf_file):
        imergf_ds = xr.open_dataset(config.imergf_file, decode_times=True, engine=NETCDF_ENGINE)
        if imergf_ds.lat[0] > imergf_ds.lat[-1]:
            imergf_ds = imergf_ds.reindex(lat=imergf_ds.lat[::-1])
        ref_datasets['imergf'] = imergf_ds
    else:
        logging.warning(f"IMERGF file not found: {config.imergf_file} - skipping IMERGF combos")

    for label, ds in ref_datasets.items():
        logging.info(f"  {label}: {ds.time.values[0]} to {ds.time.values[-1]}")

    # --- Build combos ---
    combos = build_metric_combos(month_str, dekad_str)

    # --- Iterate combos ---
    output_files = []
    for i, combo in enumerate(combos, 1):
        ref_label = combo['ref_label']
        test_label = combo['test_label']
        test_file = combo['test_file']
        out_dir = combo['output_dir']

        # Skip if reference dataset not available
        if ref_label not in ref_datasets:
            logging.info(f"[{i}/9] Skipping {ref_label} vs {test_label} - reference not loaded")
            output_files.append(None)
            continue

        # Skip if test file doesn't exist
        if not os.path.isfile(test_file):
            logging.info(f"[{i}/9] Skipping {ref_label} vs {test_label} - test file not found: {test_file}")
            output_files.append(None)
            continue

        logging.info(f"[{i}/9] {ref_label} vs {test_label}")

        # Load test dataset
        test_ds = xr.open_dataset(test_file, decode_times=True, engine=NETCDF_ENGINE)
        if test_ds.lat[0] > test_ds.lat[-1]:
            test_ds = test_ds.reindex(lat=test_ds.lat[::-1])

        # Get reference dataset (regrid CPC if needed)
        if ref_label == 'cpc':
            ref_ds = unify_cpc_for_metrics(ref_datasets['cpc'], test_ds)
            if not ref_ds.data_vars:
                logging.warning(f"  No overlap after CPC regridding - skipping")
                test_ds.close()
                output_files.append(None)
                continue
            varname_ref = CPC_PRECIP_VAR
        else:
            ref_ds = ref_datasets[ref_label]
            varname_ref = IMERG_PRECIP_VAR

        varname_test = IMERG_PRECIP_VAR

        # Slice month/dekad
        ref_slice = slice_month_dekad(ref_ds[varname_ref], month, dekad)
        test_slice = slice_month_dekad(test_ds[varname_test], month, dekad)

        # Apply land-sea mask
        ref_slice = apply_land_sea_mask(ref_slice, config.mask_file)
        test_slice = apply_land_sea_mask(test_slice, config.mask_file)

        # Compute metrics
        if mode == 'timeseries':
            metrics_ds = compute_dekad_metrics_timeseries(ref_slice, test_slice)
        else:
            metrics_ds = compute_single_dekad_metrics(ref_slice, test_slice)

        # Save
        out_fname = f"{config.FILENAME_PREFIX}_{prefix}_{ref_label}_{test_label}_month{month_str}_dekad{dekad_str}.nc4"
        os.makedirs(out_dir, exist_ok=True)
        out_fpath = os.path.join(out_dir, out_fname)

        desc = f"{ref_label.upper()} vs {test_label.upper()} - {mode} metrics"
        result = save_metrics(metrics_ds, out_fpath, description=desc)
        output_files.append(result)

        test_ds.close()
        logging.info(f"  Done: {out_fname}")

    # Close reference datasets
    for ds in ref_datasets.values():
        ds.close()

    n_saved = sum(1 for f in output_files if f is not None)
    logging.info(f"Pipeline complete: {n_saved}/{len(combos)} files saved ({mode} mode)")

    return output_files
