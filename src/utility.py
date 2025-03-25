"""
Module: utility.py

This module provides utility functions used throughout the bias correction workflow.
It includes functions for:
  - Prompting the user for a decision when an output file already exists.
  - Applying a land-sea mask to xarray datasets.
  - Ensuring that the time index of a dataset is strictly monotonic and free of duplicates.
  - Reindexing and aligning datasets with a reference dataset.

These functions help prepare data and manage I/O operations in the overall workflow.

**Author**:
Benny Istanto
- GOST/DECSC/DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
- Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import xarray as xr
import pandas as pd
import logging

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# Global variable to store user decision if a file already exists
user_choice = None

# User decision on existing files
def set_user_decision():
    """
    Prompt the user for a decision when an output file already exists and return the decision.

    Returns:
      str: The user's decision - 'O' for Overwrite, 'S' for Skip, or 'A' for Abort.
    """
    global user_choice
    if user_choice is None:
        decision = input(
            "An output file already exists. Do you want to Overwrite (O), Skip (S), or Abort (A): "
        ).upper()
        while decision not in ['O', 'S', 'A']:
            logging.info("Invalid choice. Please choose again.")
            decision = input(
                "Choose an action - Overwrite (O), Skip (S), Abort (A): "
            ).upper()
        user_choice = decision
    return user_choice

# ----
# Apply the mask to take out the sea
def apply_land_sea_mask(
        data,
        mask_file
    ):
    """
    Apply the land-sea mask to the input dataset.

    Parameters:
    data (xarray.DataArray or xarray.Dataset): The data to which the land-sea mask should be applied.
    mask_file (str): Path to the NetCDF file containing the land-sea mask.

    Returns:
    xarray.DataArray or xarray.Dataset: The data with the land-sea mask applied, keeping only land areas.
    """
    # Load the land-sea mask from the external NetCDF file
    mask_ds = xr.open_dataset(mask_file)

    # Create a boolean mask from the 'land' variable
    land_sea_mask = mask_ds['land']

    # Log ranges and shapes
    logging.info(f"Data shape: {data.shape}, Mask shape: {land_sea_mask.shape}")
    logging.info(f"Data lat range: {data.lat.min().values} to {data.lat.max().values}")
    logging.info(f"Mask lat range: {land_sea_mask.lat.min().values} to {land_sea_mask.lat.max().values}")
    logging.info(f"Data lon range: {data.lon.min().values} to {data.lon.max().values}")
    logging.info(f"Mask lon range: {land_sea_mask.lon.min().values} to {land_sea_mask.lon.max().values}")

    # Interpolate the mask to match data resolution
    land_sea_mask_reindexed = land_sea_mask.interp(lat=data.lat, lon=data.lon, method="nearest")

    # Log interpolated mask shape
    logging.info(f"Reindexed mask shape: {land_sea_mask_reindexed.shape}")

    # Apply the mask
    masked_data = data.where(land_sea_mask_reindexed == 1, drop=True)

    # Log resulting data shape
    logging.info(f"Masked data shape: {masked_data.shape}")

    mask_ds.close()

    return masked_data

# ----
# Ensure time index is strictly monotonic, sorted, and duplicates are removed
def ensure_strict_monotonic_time(
        ds
    ):
    """
    Ensure the time index in the dataset is strictly monotonic, sorted, and duplicates are removed.

    Parameters:
    ds (xarray.Dataset): The dataset to process.

    Returns:
    xarray.Dataset: The dataset with a cleaned and strictly monotonic time index.
    """
    # Sort by time to ensure monotonicity
    ds = ds.sortby('time')

    # Remove any duplicate timestamps
    ds = ds.sel(time=~ds.get_index("time").duplicated())

    # Ensure strict monotonicity by dropping non-monotonic entries
    time_diff = ds['time'].diff('time')
    non_monotonic = time_diff <= pd.Timedelta(0)

    if non_monotonic.any():
        logging.warning(f"Found non-monotonic time steps: {ds['time'].where(non_monotonic, drop=True).values}")
        ds = ds.sel(time=~non_monotonic)

    return ds

# ----
# Reindex and align datasets while ensuring strict monotonicity
def reindex_and_align_with_monotonicity(
        reference_ds,
        secondary_ds,
        land_sea_mask
    ):
    """
    Reindex and align the secondary dataset with a reference dataset while ensuring
    the reference dataset has a strictly monotonic, duplicate-free time index.

    Parameters
    ----------
    reference_ds : xarray.Dataset
        The reference dataset (e.g., IMERG) to which we'll align.
    secondary_ds : xarray.Dataset
        The secondary dataset (e.g., CPC) that needs alignment.
    land_sea_mask : xarray.DataArray
        Land-sea mask for spatial alignment.

    Returns
    -------
    tuple(xarray.Dataset, xarray.DataArray)
        The aligned secondary dataset and a spatially aligned land-sea mask.
    """
    # Ensure strict monotonicity in time for both datasets
    reference_ds = ensure_strict_monotonic_time(reference_ds)
    secondary_ds = ensure_strict_monotonic_time(secondary_ds)

    # Reindex and align secondary dataset to the reference dataset
    secondary_ds_aligned = secondary_ds.reindex_like(reference_ds, method='nearest')

    # Align land-sea mask spatially with the reference dataset
    land_sea_mask_aligned = land_sea_mask.interp(lat=reference_ds.lat, lon=reference_ds.lon, method='nearest')

    return secondary_ds_aligned, land_sea_mask_aligned
