"""
Module: bias_correction.py

This module provides the high-level bias correction workflow for daily precipitation data.
It combines Linear Scaling (LS) and Empirical Quantile Mapping (EQM) with GPD tail adjustment
to correct bias in IMERG data using CPC data as reference.

The main function, lseqm, performs the following steps:
  1. Validates input datasets.
  2. Aggregates multi-year IMERG and CPC data for the specified dekad.
  3. Aligns the aggregated datasets.
  4. Applies LS by scaling the IMERG data with the ratio of CPC to IMERG means.
  5. Applies EQM via gamma quantile mapping with GPD tail adjustment.
  6. Saves intermediate LS and LSEQM products if requested.
  7. Returns the LSEQM-corrected data along with the CPC dekad data
     (needed as training target for the optional DL refinement step).

The Deep Learning (DL) refinement is handled separately in deep_learning.py.
This two-step design eliminates domain shift: the DL model is trained on
LSEQM-corrected data (not raw IMERG), matching what it receives at inference.

**Author**:
  Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import xarray as xr
import logging
import numpy as np
from .io import save_corrected_precip, get_max_day_in_month, aggregate_data_across_years
from .distribution_fitting import gamma_quantile_mapping

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# ----
# LSEQM method for bias correction
def lseqm(
        imerg_ds,
        cpc_ds,
        month,
        dekad_start_day,
        dekad_end_day,
        save_ls_result=True,
        save_lseqm_result=True,
        month_str=None,
        dekad_str=None,
        ls_corrected_precip_path=None,
        lseqm_corrected_precip_path=None
    ):
    """
    Apply Linear Scaling (LS) and Empirical Quantile Mapping (EQM) with GPD tail
    adjustment for bias correction of daily precipitation data, using data aggregated
    across years for the specified dekad.

    This is Step 1 of the two-step workflow. The returned LSEQM result and CPC dekad
    data can be passed to train_bias_correction_model() and apply_deeplearning_model()
    for the optional DL refinement (Step 2).

    Parameters:
    ----------
    imerg_ds : xarray.Dataset
        IMERG precipitation dataset with dimensions ('time', 'lat', 'lon').
    cpc_ds : xarray.Dataset
        CPC precipitation dataset with dimensions ('time', 'lat', 'lon').
    month : int
        The month number (1-12) for which the dekad is specified.
    dekad_start_day : int
        Start day of the dekad (e.g., 1, 11, 21).
    dekad_end_day : int
        End day of the dekad (e.g., 10, 20, last day of month).
    save_ls_result : bool, optional
        If True, saves the Linear Scaling (LS) corrected precipitation data. Default is True.
    save_lseqm_result : bool, optional
        If True, saves the LSEQM corrected precipitation data. Default is True.
    month_str : str, optional
        Two-digit string representing the month (e.g., '01', '02', ..., '12').
    dekad_str : str, optional
        String representing the dekad (e.g., '01', '11', '21').
    ls_corrected_precip_path : str, optional
        Directory where LS-corrected data will be saved.
    lseqm_corrected_precip_path : str, optional
        Directory where LSEQM-corrected data will be saved.

    Returns:
    ----------
    tuple of (xarray.DataArray, xarray.DataArray)
        (lseqm_corrected_precip, cpc_dekad_data)
        The LSEQM-corrected precipitation and the aggregated CPC dekad data.
        The CPC dekad data is returned so it can be used as the training target
        for the DL refinement step.
    """
    # Ensure that month_str and dekad_str are provided
    if month_str is None or dekad_str is None:
        raise ValueError("month_str and dekad_str must be provided.")

    # Add data validation at the start (xarray-compatible NaN check).
    # Check only the precipitation variable — the land-sea mask sets ocean
    # pixels to NaN, so checking the whole Dataset would wrongly trigger this.
    from .config import IMERG_PRECIP_VAR, CPC_PRECIP_VAR
    _imerg_var = imerg_ds[IMERG_PRECIP_VAR] if isinstance(imerg_ds, xr.Dataset) else imerg_ds
    _cpc_var = cpc_ds[CPC_PRECIP_VAR] if isinstance(cpc_ds, xr.Dataset) else cpc_ds
    if _imerg_var.isnull().all().item() or _cpc_var.isnull().all().item():
        logging.error("Invalid input data - all NaN values in precipitation variable")
        raise ValueError("Invalid input data")

    # Aggregate data across all years for the specified dekad
    imerg_dekad_data, cpc_dekad_data = aggregate_data_across_years(imerg_ds, cpc_ds, month, dekad_start_day, dekad_end_day)
    logging.info(f"IMERG dekad data shape: {imerg_dekad_data.shape}")
    logging.info(f"CPC dekad data shape: {cpc_dekad_data.shape}")

    # Add validation after aggregation
    if imerg_dekad_data.size == 0 or cpc_dekad_data.size == 0:
        logging.error("No data available after aggregation")
        raise ValueError("No data available for correction")

    # Ensure data alignment
    imerg_dekad_data, cpc_dekad_data = xr.align(imerg_dekad_data, cpc_dekad_data, join='inner')

    # Perform Linear Scaling (LS)
    logging.info("Performing Linear Scaling (LS)...")
    imerg_mean = imerg_dekad_data.mean(dim='time')
    cpc_mean = cpc_dekad_data.mean(dim='time')

    ls_scale_factor = xr.where(
        imerg_mean != 0,
        cpc_mean / imerg_mean,
        1
    )

    # Apply LS scaling to IMERG data
    ls_corrected_precip = imerg_dekad_data * ls_scale_factor

    # Save LS result if requested
    if save_ls_result:
        logging.info("Saving LS corrected precipitation...")
        save_corrected_precip(
            ls_corrected_precip,
            imerg_dekad_data,
            method_abbr="ls",
            method_full="Linear Scaling",
            folder=ls_corrected_precip_path,
            dekad_str=dekad_str,
            month_str=month_str
        )

    # Perform Empirical Quantile Mapping (EQM)
    import time as _time
    n_lat = len(ls_corrected_precip.lat)
    n_lon = len(ls_corrected_precip.lon)
    n_time = len(ls_corrected_precip.time)
    logging.info(f"Applying Empirical Quantile Mapping (EQM) on {n_lat}x{n_lon} grid...")

    eqm_data = np.full_like(ls_corrected_precip.values, np.nan)
    _t0 = _time.time()
    _land_count = 0

    for i in range(n_lat):
        for j in range(n_lon):
            imerg_ts = ls_corrected_precip.values[:, i, j]
            cpc_ts = cpc_dekad_data.values[:, i, j]
            # Skip all-NaN pixels (ocean) without calling the function
            if np.all(np.isnan(imerg_ts)) or np.all(np.isnan(cpc_ts)):
                continue
            _land_count += 1
            eqm_data[:, i, j] = gamma_quantile_mapping(imerg_ts, cpc_ts)

        # Progress every 10 rows
        if (i + 1) % 10 == 0 or (i + 1) == n_lat:
            elapsed = _time.time() - _t0
            pct = (i + 1) / n_lat * 100
            eta = elapsed / (i + 1) * (n_lat - i - 1) if i > 0 else 0
            logging.info(f"  EQM progress: row {i+1}/{n_lat} ({pct:.0f}%) "
                         f"| {_land_count} land pixels done "
                         f"| elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    eqm_corrected_precip = xr.DataArray(
        eqm_data,
        coords=ls_corrected_precip.coords,
        dims=ls_corrected_precip.dims,
        attrs=ls_corrected_precip.attrs,
    )

    # Ensure non-negative precipitation values
    eqm_corrected_precip = eqm_corrected_precip.clip(min=0)

    # Log summary of quantile mapping coverage
    _total_px = int(np.prod(eqm_corrected_precip.isel(time=0).shape))
    _valid_px = int((~eqm_corrected_precip.isel(time=0).isnull()).sum().item())
    logging.info(f"EQM complete: {_valid_px}/{_total_px} grid points corrected "
                 f"({_total_px - _valid_px} masked/ocean pixels skipped)")

    # Save LSEQM result if requested
    if save_lseqm_result:
        logging.info("Saving LSEQM corrected precipitation...")
        save_corrected_precip(
            eqm_corrected_precip,
            imerg_dekad_data,
            method_abbr="lseqm",
            method_full="Linear Scaling and Empirical Quantile Mapping",
            folder=lseqm_corrected_precip_path,
            dekad_str=dekad_str,
            month_str=month_str
        )

    return eqm_corrected_precip, cpc_dekad_data
