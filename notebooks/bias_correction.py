"""
Module: bias_correction.py

This module provides the high-level bias correction workflow for daily precipitation data.
It combines Linear Scaling (LS) and Empirical Quantile Mapping (EQM) with an optional Deep Learning (DL)
adjustment to correct bias in IMERG data using CPC data as reference.

The main function, lseqmdf, performs the following steps:
  1. Validates input datasets.
  2. Aggregates multi-year IMERG and CPC data for the specified dekad.
  3. Aligns the aggregated datasets.
  4. Applies LS by scaling the IMERG data with the ratio of CPC to IMERG means.
  5. Applies EQM via gamma quantile mapping with tail adjustment.
  6. Optionally applies a trained DL model to further correct extreme values.
  7. Saves intermediate and final products if requested.

Output directories for LS, LSEQM, and DL-corrected products are passed as parameters,
allowing for flexible configuration.

Author:
  Benny Istanto
  - GOST/DECSC/DEC Data Group, The World Bank, United States (bistanto@worldbank.org)
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia (bistanto@ipb.ac.id)
with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa
Update: 2025
"""
# Import the library
import xarray as xr
import logging
import numpy as np
from .io import aggregate_data_across_years, save_corrected_precip
from .io import get_max_day_in_month
from .utility import gamma_quantile_mapping
from .deep_learning import apply_deeplearning_model

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# ----
# LSEQM+DL method for bias correction
def lseqmdf(
        imerg_ds,
        cpc_ds,
        month,
        dekad_start_day,
        dekad_end_day,
        method_abbr="lseqm",
        method_full="Linear Scaling and Empirical Quantile Mapping",
        model=None,
        save_ls_result=True,
        save_lseqm_result=True,
        save_dl_result=True,
        month_str=None,
        dekad_str=None, 
        ls_corrected_precip_path=None, 
        lseqm_corrected_precip_path=None, 
        lseqmdl_corrected_precip_path=None
    ):
    """
    Apply Linear Scaling (LS) and Empirical Quantile Mapping (EQM) for bias correction of daily precipitation data,
    using data aggregated across years for the specified dekad. Optionally, apply a Deep Learning (DL) model for further
    correction on extreme values.

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
    method_abbr : str, optional
        Abbreviation for the correction method (e.g., 'ls', 'lseqm'). Default is "lseqm".
    method_full : str, optional
        Full name of the correction method for file metadata. Default is "Linear Scaling and Empirical Quantile Mapping".
    model : object, optional
        Trained deep learning model for bias correction. If None, DL-based adjustments are skipped. Default is None.
    save_ls_result : bool, optional
        If True, saves the Linear Scaling (LS) corrected precipitation data. Default is True.
    save_lseqm_result : bool, optional
        If True, saves the LSEQM corrected precipitation data. Default is True.
    save_dl_result : bool, optional
        If True, saves the DL-corrected precipitation data. Default is True.
    month_str : str, optional
        Two-digit string representing the month (e.g., '01', '02', ..., '12').
    dekad_str : str, optional
        String representing the dekad (e.g., '01', '11', '21').
    ls_corrected_precip_path : str, optional
        Directory where LS-corrected data will be saved.
    lseqm_corrected_precip_path : str, optional
        Directory where LSEQM-corrected data will be saved.
    lseqmdl_corrected_precip_path : str, optional
        Directory where DL-corrected data will be saved.

    Returns:
    ----------
    xarray.DataArray
        The final bias-corrected precipitation data for the specified dekad.
    """
    # Ensure that month_str and dekad_str are provided
    if month_str is None or dekad_str is None:
        raise ValueError("month_str and dekad_str must be provided.")

    # Add data validation at the start
    if np.all(np.isnan(imerg_ds)) or np.all(np.isnan(cpc_ds)):
        logging.error("Invalid input data - all NaN values")
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
    logging.info("Applying Empirical Quantile Mapping (EQM)...")
    # Apply gamma quantile mapping
    eqm_corrected_precip = xr.apply_ufunc(
        gamma_quantile_mapping,
        ls_corrected_precip,
        cpc_dekad_data,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        vectorize=True,
        output_dtypes=[ls_corrected_precip.dtype],
        keep_attrs=True
    ).compute()

    # Ensure non-negative precipitation values
    eqm_corrected_precip = eqm_corrected_precip.clip(min=0)

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

    # Apply DL model for bias correction if provided
    if model is not None:
        logging.info("Applying DL model for bias correction...")
        # Prepare input data for the model
        corrected_precip = apply_deeplearning_model(model, eqm_corrected_precip)

        # Ensure non-negative precipitation values
        corrected_precip = corrected_precip.clip(min=0)

        # Save DL corrected precipitation
        if save_dl_result:
            logging.info("Saving DL corrected precipitation...")
            save_corrected_precip(
                corrected_precip,
                eqm_corrected_precip,
                method_abbr="lseqmdl",
                method_full="Hybrid Deep Learning-Physical (Linear Scaling and Empirical Quantile Mapping) Approach",
                folder=lseqmdl_corrected_precip_path,
                dekad_str=dekad_str,
                month_str=month_str
            )

        return corrected_precip
    else:
        return eqm_corrected_precip
