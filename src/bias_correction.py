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

Update: 2026.03
"""
# Import the library
import xarray as xr
import logging
import numpy as np
from .io import save_corrected_precip, get_max_day_in_month, aggregate_data_across_years, aggregate_cpc_native_for_dekad
from .distribution_fitting import (
    gamma_quantile_mapping,
    gamma_quantile_mapping_precomputed,
    fit_cpc_parameters_on_native_grid,
    interpolate_cpc_params_to_imerg_grid,
)

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
        lseqm_corrected_precip_path=None,
        cpc_native_ds=None
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
        CPC precipitation dataset with dimensions ('time', 'lat', 'lon'),
        already aligned (reindexed) to the IMERG grid.
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
    cpc_native_ds : xarray.Dataset, optional
        CPC dataset at native ~0.5° resolution (before regridding). When provided,
        CPC distribution parameters are fitted at native resolution and bilinearly
        interpolated to the IMERG grid, eliminating the 0.5° block boundary artefact.
        When None, the original per-pixel fitting is used (backward compatible).

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

    # --- Native CPC parameter fitting (Option B) ---
    _use_native_cpc = cpc_native_ds is not None
    interp_cpc_params = None
    if _use_native_cpc:
        logging.info("Using native-resolution CPC parameter fitting (BCSD principle)...")
        cpc_native_dekad = aggregate_cpc_native_for_dekad(
            cpc_native_ds, month, dekad_start_day, dekad_end_day
        )
        cpc_params = fit_cpc_parameters_on_native_grid(cpc_native_dekad)
        interp_cpc_params = interpolate_cpc_params_to_imerg_grid(
            cpc_params,
            imerg_dekad_data.lat.values,
            imerg_dekad_data.lon.values
        )

    # Perform Linear Scaling (LS)
    logging.info("Performing Linear Scaling (LS)...")
    imerg_mean = imerg_dekad_data.mean(dim='time')

    if _use_native_cpc:
        # Smooth LS: bilinearly interpolate CPC mean from native resolution
        cpc_native_mean = cpc_native_dekad.mean(dim='time')
        cpc_mean = cpc_native_mean.interp(
            lat=imerg_dekad_data.lat, lon=imerg_dekad_data.lon, method='linear'
        )
        # Fill boundary NaN with nearest-neighbour
        cpc_mean = cpc_mean.fillna(
            cpc_native_mean.interp(
                lat=imerg_dekad_data.lat, lon=imerg_dekad_data.lon, method='nearest'
            )
        )
        logging.info("LS using bilinearly interpolated CPC mean (smooth).")
    else:
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

            # Skip all-NaN IMERG pixels (ocean)
            if np.all(np.isnan(imerg_ts)):
                continue

            if _use_native_cpc:
                # Use pre-computed, smoothly interpolated CPC parameters
                _land_count += 1
                eqm_data[:, i, j] = gamma_quantile_mapping_precomputed(
                    imerg_ts,
                    float(interp_cpc_params['gamma_shape'].values[i, j]),
                    float(interp_cpc_params['gamma_scale'].values[i, j]),
                    float(interp_cpc_params['gpd_threshold'].values[i, j]),
                    float(interp_cpc_params['gpd_shape'].values[i, j]),
                    float(interp_cpc_params['gpd_loc'].values[i, j]),
                    float(interp_cpc_params['gpd_scale'].values[i, j]),
                    float(interp_cpc_params['upper_cap'].values[i, j]),
                    float(interp_cpc_params['p_threshold'].values[i, j]),
                )
            else:
                # Original per-pixel fitting path (backward compatible)
                cpc_ts = cpc_dekad_data.values[:, i, j]
                if np.all(np.isnan(cpc_ts)):
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


# ----
# Full correction pipeline: LS → LSEQM → DL (Steps 7-10 of notebook 02)
def run_correction_pipeline(imerg_ds, cpc_ds_aligned, month, dekad, cpc_native_ds=None):
    """
    Run the full LS → LSEQM → LSEQMDL pipeline for one month/dekad period.

    This wraps Steps 7–10 of notebook 02 into a single callable, designed
    for batch processing where ``interactive=False`` is used so that existing
    DL models are reloaded silently without prompting.

    Parameters
    ----------
    imerg_ds : xarray.Dataset
        Full IMERG precipitation dataset (all years, all months).
    cpc_ds_aligned : xarray.Dataset
        CPC dataset already aligned (reindexed) to the IMERG grid.
    month : int
        Month number (1–12).
    dekad : int
        Dekad number (1, 2, or 3).
    cpc_native_ds : xarray.Dataset, optional
        CPC dataset at native ~0.5° resolution (before regridding). When provided,
        enables native-resolution CPC parameter fitting to eliminate the 0.5° block
        boundary artefact. When None, uses the original per-pixel fitting.

    Returns
    -------
    str
        Path to the saved LSEQMDL-corrected NetCDF file.
    """
    import os
    from . import config
    from .deep_learning import train_bias_correction_model, apply_deeplearning_model

    # Derive dekad variables
    month_str = f"{month:02d}"
    dekad_str = '01' if dekad == 1 else ('11' if dekad == 2 else '21')
    dekad_start = int(dekad_str)
    dekad_end = (
        10 if dekad == 1
        else 20 if dekad == 2
        else get_max_day_in_month(imerg_ds, month)
    )

    # Step 7: LSEQM bias correction (LS + EQM + GPD)
    logging.info("Pipeline [%s d%s]: Running LSEQM...", month_str, dekad)
    lseqm_result, cpc_dekad_data = lseqm(
        imerg_ds, cpc_ds_aligned, month, dekad_start, dekad_end,
        month_str=month_str,
        dekad_str=dekad_str,
        ls_corrected_precip_path=config.ls_corrected_precip_path,
        lseqm_corrected_precip_path=config.lseqm_corrected_precip_path,
        cpc_native_ds=cpc_native_ds,
    )

    # Step 8: Train (or reload) DL model — interactive=False for batch
    model_name = f"bias_correction_model_month{month_str}_dekad{dekad_str}"
    logging.info("Pipeline [%s d%s]: Training / loading DL model...", month_str, dekad)
    model = train_bias_correction_model(
        lseqm_result, cpc_dekad_data, model_name,
        interactive=False,
    )

    # Step 9: Confidence mask (optional, controlled by config)
    confidence_mask = None
    if (config.USE_CONFIDENCE_MASK
            and config.STATION_FILE
            and os.path.isfile(config.STATION_FILE)):
        from .station_density import get_or_create_confidence_mask
        logging.info("Pipeline [%s d%s]: Loading confidence mask...", month_str, dekad)
        confidence_mask = get_or_create_confidence_mask(
            station_file=config.STATION_FILE,
            confidence_mask_file=config.CONFIDENCE_MASK_FILE,
            target_lat=lseqm_result.lat.values,
            target_lon=lseqm_result.lon.values,
            cpc_resolution=config.DENSITY_CPC_RESOLUTION,
            smoothing_sigma=config.DENSITY_SMOOTHING_SIGMA,
            saturation_count=config.DENSITY_SATURATION_COUNT,
            lat_range=config.DENSITY_LAT_RANGE,
            lon_range=config.DENSITY_LON_RANGE,
        )

    # Step 10: Apply DL refinement + save
    logging.info("Pipeline [%s d%s]: Applying DL refinement...", month_str, dekad)
    corrected_precip = apply_deeplearning_model(
        model, lseqm_result, confidence_mask=confidence_mask
    ).clip(min=0)

    save_corrected_precip(
        corrected_precip,
        lseqm_result,
        method_abbr="lseqmdl",
        method_full=(
            "Hybrid Deep Learning-Physical "
            "(Linear Scaling and Empirical Quantile Mapping) Approach"
        ),
        folder=config.lseqmdl_corrected_precip_path,
        dekad_str=dekad_str,
        month_str=month_str,
    )

    logging.info("Pipeline [%s d%s]: Complete.", month_str, dekad)

    out_fname = (
        f"{config.FILENAME_PREFIX}_imergl_lseqmdl_corrected_precip"
        f"_month{month_str}_dekad{dekad_str}.nc4"
    )
    return os.path.join(config.lseqmdl_corrected_precip_path, out_fname)
