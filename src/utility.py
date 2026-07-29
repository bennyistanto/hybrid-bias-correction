"""
Module: utility.py

This module provides utility functions used throughout the bias correction workflow.
It includes functions for:
  - Prompting the user for a decision when an output file already exists (with non-interactive support).
  - Loading and caching the land-sea mask to avoid repeated I/O.
  - Applying a land-sea mask to xarray datasets.
  - Ensuring that the time index of a dataset is strictly monotonic and free of duplicates.
  - Reindexing and aligning datasets with a reference dataset.

These functions help prepare data and manage I/O operations in the overall workflow.

**Author**:
  Benny Istanto
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.07
"""
# Import the library
import xarray as xr
import pandas as pd
import logging

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# Global variable to store user decision if a file already exists
_user_choice = None

# Cache for the land-sea mask to avoid repeated file I/O
_mask_cache = {}


def reset_user_choice():
    """Reset the stored user choice (useful when processing a new batch)."""
    global _user_choice
    _user_choice = None


# User decision on existing files
def set_user_decision(interactive=None):
    """
    Get the user's decision when an output file already exists.

    In interactive mode, prompts the user once and remembers the choice for the session.
    In non-interactive mode, returns the default action from config.

    Parameters:
    ----------
    interactive : bool, optional
        Override the config INTERACTIVE setting. If None, uses config value.

    Returns:
    ----------
    str
        The user's decision - 'O' for Overwrite, 'S' for Skip, or 'A' for Abort.
    """
    global _user_choice

    # Determine interactive mode
    if interactive is None:
        from .config import INTERACTIVE
        interactive = INTERACTIVE

    if _user_choice is not None:
        return _user_choice

    if not interactive:
        # Non-interactive: use default from config
        from .config import EXISTING_FILE_ACTION
        action_map = {
            'overwrite': 'O',
            'skip': 'S',
            'abort': 'A'
        }
        _user_choice = action_map.get(EXISTING_FILE_ACTION.lower(), 'S')
        logging.info(f"Non-interactive mode: using '{EXISTING_FILE_ACTION}' for existing files")
        return _user_choice

    # Interactive: prompt the user
    decision = input(
        "An output file already exists. Do you want to Overwrite (O), Skip (S), or Abort (A): "
    ).upper()
    while decision not in ['O', 'S', 'A']:
        logging.info("Invalid choice. Please choose again.")
        decision = input(
            "Choose an action - Overwrite (O), Skip (S), Abort (A): "
        ).upper()
    _user_choice = decision
    return _user_choice


# ----
# Load and cache the land-sea mask
def load_mask(mask_file, mask_var=None):
    """
    Load the land-sea mask from a NetCDF file, with caching to avoid repeated I/O.

    The mask is loaded once per unique file path and cached for subsequent calls.

    Parameters:
    ----------
    mask_file : str
        Path to the NetCDF file containing the land-sea mask.
    mask_var : str, optional
        Variable name for the land mask. If None, uses config.MASK_VAR.

    Returns:
    ----------
    xarray.DataArray
        The land-sea mask (1 = land, 0 = sea/ocean).
    """
    if mask_var is None:
        from .config import MASK_VAR
        mask_var = MASK_VAR

    if mask_file not in _mask_cache:
        logging.info(f"Loading land-sea mask from {mask_file} (will be cached)")
        from .config import NETCDF_ENGINE
        with xr.open_dataset(mask_file, engine=NETCDF_ENGINE) as mask_ds:
            # Load into memory so the file handle can be closed
            _mask_cache[mask_file] = mask_ds[mask_var].load()
    return _mask_cache[mask_file]


# ----
# Apply the mask to take out the sea
def apply_land_sea_mask(
        data,
        mask_file,
        mask_var=None
    ):
    """
    Apply the land-sea mask to the input dataset.

    Uses cached mask to avoid repeated file I/O. The mask is interpolated
    (nearest-neighbor) to match the data resolution if needed.

    Parameters:
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to which the land-sea mask should be applied.
    mask_file : str
        Path to the NetCDF file containing the land-sea mask.
    mask_var : str, optional
        Variable name for the land mask. If None, uses config.MASK_VAR.

    Returns:
    ----------
    xarray.DataArray or xarray.Dataset
        The data with the land-sea mask applied (ocean pixels set to NaN).
    """
    # Load mask from cache
    land_sea_mask = load_mask(mask_file, mask_var)

    # Align the mask to the data grid. Use reindex(method='nearest') rather
    # than interp(method='nearest'): reindex is a pure lookup tolerant of
    # small numerical differences in coordinate values, while interp goes
    # through scipy and returns NaN for any target outside the mask's
    # coordinate range. The latter silently drops the southernmost row and
    # westernmost column when data and mask coords differ in dtype (e.g.
    # data is float32, mask is float64), because the float32 representation
    # of the boundary value is slightly outside the float64 range.
    land_sea_mask_reindexed = land_sea_mask.reindex(
        lat=data.lat, lon=data.lon, method="nearest"
    )

    # Apply the mask: set ocean pixels to NaN (not drop)
    # Using drop=False preserves the grid structure while marking ocean as NaN
    masked_data = data.where(land_sea_mask_reindexed == 1)

    return masked_data

# ----
# Ensure time index is strictly monotonic, sorted, and duplicates are removed
def ensure_strict_monotonic_time(
        ds
    ):
    """
    Ensure the time index in the dataset is strictly monotonic, sorted, and duplicates are removed.

    Parameters:
    ----------
    ds : xarray.Dataset
        The dataset to process.

    Returns:
    ----------
    xarray.Dataset
        The dataset with a cleaned and strictly monotonic time index.
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
