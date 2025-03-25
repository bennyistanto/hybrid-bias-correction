"""
Module: io.py

This module provides functions for file input/output operations for the bias correction workflow.
It includes:
  - save_corrected_precip: Saves corrected precipitation data to a NetCDF file following CF metadata conventions.
  - get_max_day_in_month: Finds the maximum day in a given month (accounting for leap years).
  - aggregate_data_across_years: Aggregates IMERG and CPC data for a specified dekad over all years.

It also uses utility functions for applying a land-sea mask and prompting the user for decisions.

**Author**:
Benny Istanto
- GOST/DECSC/DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
- Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import os
import xarray as xr
import numpy as np
import pandas as pd
import calendar
import logging
from .config import cf18_f32, mask_file, output_filename_template
from .utility import apply_land_sea_mask, set_user_decision

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# ----
# Helper function to save precipitation data to NetCDF with consistent metadata
def save_corrected_precip(
        precip_data,
        ds,
        method_abbr,
        method_full,
        folder,
        dekad_str,
        month_str
    ):
    """
    Save precipitation data to NetCDF with metadata and proper filename formatting.

    Parameters:
    precip_data (xarray.DataArray): Precipitation data to be saved.
    ds (xarray.Dataset): Original dataset for coordinates and attributes.
    method_abbr (str): Abbreviation of the method (e.g., 'ls', 'lseqm', 'lseqmdl').
    method_full (str): Full name of the method (e.g., 'Linear Scaling', 'LSEQM').
    folder (str): Directory where the corrected precipitation will be saved.
    dekad_str (str): String representing the dekad (e.g., '01', '11', '21').
    month_str (str):  Two-digit string representing the month (e.g., '01', '02', ..., '12').

    Returns:
    -------
    str or None
        Path to the saved file, or None if saving failed.
    """
    # Generate output filename
    output_file = output_filename_template.format(
        folder=folder,
        method_abbr=method_abbr,
        month_str=month_str,
        dekad_str=dekad_str
    )

    # Check if output file exists
    if os.path.exists(output_file):
        logging.info(f"File {output_file} already exists.")
        decision = set_user_decision()  # Now returns the decision.

        if decision == 'S':
            logging.info(f"Skipping file {output_file}")
            return  # Skip saving
        elif decision == 'A':
            logging.info("Aborting process.")
            raise SystemExit  # Abort the process if the user decides to stop
        elif decision == 'O':
            logging.info(f"Overwriting file {output_file}")

    logging.info(f"Precip data dims: {precip_data.dims}, shape: {precip_data.shape}")
    # Ensure that precip_data has dimensions ('time', 'lat', 'lon')
    expected_dims = ('time', 'lat', 'lon')

    # If there's no 'time' dimension, create a dummy time dimension of length 1
    if 'time' not in precip_data.dims:
        logging.warning("Data has no 'time' dimension; adding dummy time dimension.")
        # Turn shape (lat, lon) -> (time, lat, lon)
        precip_data = precip_data.expand_dims(dim={'time': [pd.Timestamp.now()]}, axis=0)

    # Ensure lat, lon, time are in correct order
    # If user already has (time, lat, lon), no problem
    missing_dims = [d for d in expected_dims if d not in precip_data.dims]
    if missing_dims:
        logging.warning(f"Missing dims {missing_dims}, cannot reorder precisely.")
    else:
        # Reorder the dimensions
        precip_data = precip_data.transpose(*expected_dims)

    # Extract data from precip_data
    precip_values = precip_data.data

    # Create xarray Dataset for corrected precipitation
    corrected_ds = xr.Dataset(
        data_vars={
            'precipitation': (
                ('time', 'lat', 'lon'), precip_values
            )
        },
        coords={
            'time': precip_data['time'],
            'lat': precip_data['lat'],
            'lon': precip_data['lon']
        },
        # Below information will appear as metadata in the output file
        # Feel free to adjust or modify, especially on the creator name, role and email
        attrs={
            'cdm_data_type': 'GRID',
            'title': f'Bias Corrected IMERG Late Precipitation using {method_full}',
            'summary': f'Precipitation data corrected using {method_full}',
            'source': 'IMERG and CPC-UNI',
            'history': f'Created on {pd.Timestamp.now()}',
            'DOI': '10.5067/GPM/IMERGDL/DAY/07',
            'creator_name': 'Benny Istanto',
            'creator_role': 'Climate Geographer',
            'creator_email': 'bistanto@worldbank.org',
            'comment': f'This dataset has been bias corrected using {method_full}'
        }
    )

    # Update metadata attributes
    corrected_ds['precipitation'].attrs.update({
        'units': 'mm',
        'long_name': 'Corrected daily mean precipitation rate estimate',
        'standard_name': 'corrected_precipitation'
    })

    corrected_ds['lat'].attrs.update({'units': 'degrees_north', 'long_name': 'Latitude'})
    corrected_ds['lon'].attrs.update({'units': 'degrees_east', 'long_name': 'Longitude'})

    # Apply land-sea mask to the `precipitation` variable only
    masked_precip = apply_land_sea_mask(corrected_ds['precipitation'], mask_file)

    # Replace the precipitation variable in corrected_ds with the masked version:
    corrected_ds['precipitation'] = masked_precip

    # Save to NetCDF following CF Convention
    try:
        corrected_ds.to_netcdf(output_file, encoding=cf18_f32, engine='netcdf4')
        logging.info(f"Saved {method_full} corrected precipitation for month {month_str}, dekad {dekad_str} at {output_file}")
        return output_file
    except IOError as e:
        logging.error(f"Failed to save file {output_file}: {str(e)}")
        return None

# ----
# Find the maximum day in `month` across all years
def get_max_day_in_month(
        ds: xr.Dataset,
        month: int
    ) -> int:
    """
    Scan all years in the dataset `ds`, and find the maximum day
    for the specified month. For example, if month=2 (February) and
    there's at least one leap year in ds, this returns 29; otherwise 28.
    """
    unique_years = np.unique(ds['time.year'].values)
    max_day = 0
    for year in unique_years:
        days_in_this_month = calendar.monthrange(year, month)[1]  # e.g. 28 or 29 for February
        if days_in_this_month > max_day:
            max_day = days_in_this_month
    return max_day

# ----
# Aggregate IMERG and CPC data across all years for the specified dekad.
def aggregate_data_across_years(
        imerg_ds,
        cpc_ds,
        month,
        dekad_start_day,
        dekad_end_day
    ):
    """
    Aggregate IMERG and CPC data across all years for the specified dekad.

    Parameters:
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

    Returns:
    tuple of xarray.DataArray
        Aggregated IMERG and CPC data for the specified dekad across all years.
    """
    # Align datasets before applying masks
    logging.info("Aligning IMERG and CPC datasets...")
    imerg_ds, cpc_ds = xr.align(imerg_ds, cpc_ds, join="inner")

    # Create time masks
    logging.info("Creating time-based masks...")
    imerg_time_mask = (
        (imerg_ds['time.month'] == month) &
        (imerg_ds['time.day'] >= dekad_start_day) &
        (imerg_ds['time.day'] <= dekad_end_day)
    )
    cpc_time_mask = (
        (cpc_ds['time.month'] == month) &
        (cpc_ds['time.day'] >= dekad_start_day) &
        (cpc_ds['time.day'] <= dekad_end_day)
    )

    # Apply time masks
    try:
        logging.info("Applying time masks...")
        imerg_dekad_data = imerg_ds['precipitation'].where(imerg_time_mask, drop=True)
        cpc_dekad_data = cpc_ds['precip'].where(cpc_time_mask, drop=True)
    except Exception as e:
        logging.error("Error during masking:", exc_info=True)
        raise ValueError(f"Failed to apply time masks: {str(e)}")

    # Validate resulting data
    if imerg_dekad_data.size == 0 or cpc_dekad_data.size == 0:
        logging.error("No data available after masking.")
        logging.info(f"IMERG data after masking: {imerg_dekad_data.shape}")
        logging.info(f"CPC data after masking: {cpc_dekad_data.shape}")
        raise ValueError("No data found for the specified month and dekad.")

    return imerg_dekad_data, cpc_dekad_data
