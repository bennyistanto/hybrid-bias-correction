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
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.07
"""
# Import the library
import os
import xarray as xr
import numpy as np
import pandas as pd
import calendar
import logging
# NOTE: do NOT import `mask_file` at module load time - its value is set by
# initialize_config() AFTER this module imports, so a bare import would freeze
# the default Indonesia mask path. Always read it dynamically via `config.mask_file`.
from . import config as _cfg
from .config import cf18_f32, output_filename_template, IMERG_PRECIP_VAR, CPC_PRECIP_VAR, NETCDF_ENGINE
from .utility import apply_land_sea_mask, set_user_decision

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++


class BiasCorrectAbort(Exception):
    """Raised when the user chooses to abort the bias correction process."""
    pass


# ----
# Helper function to save precipitation data to NetCDF with consistent metadata
def _run_provenance():
    """Non-CF attributes recording which code and settings produced a file.

    Every lookup is defensive: provenance is a convenience, so a missing git
    binary or an uninitialised config must never break a pipeline run. Values
    are read at call time so a mid-session initialize_config() is reflected.
    """
    prov = {'run_timestamp': pd.Timestamp.now().isoformat()}

    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            prov['framework_version'] = version('hybrid-bias-correction')
        except PackageNotFoundError:
            prov['framework_version'] = 'unknown'
    except Exception:
        prov['framework_version'] = 'unknown'

    try:
        import subprocess
        prov['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        prov['git_commit'] = 'unknown'

    for attr, key in (('CONFIG_FILE_PATH', 'config_file'),
                      ('DL_BLEND_ALPHA', 'blend_alpha'),
                      ('GPD_THRESHOLD_PERCENTILE', 'gpd_threshold_percentile'),
                      ('DENSITY_SATURATION_COUNT', 'saturation_count')):
        value = getattr(_cfg, attr, None)
        if value is not None:
            prov[key] = str(value)

    return prov


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
    ----------
    precip_data : xarray.DataArray
        Precipitation data to be saved.
    ds : xarray.Dataset
        Original dataset for coordinates and attributes.
    method_abbr : str
        Abbreviation of the method (e.g., 'ls', 'lseqm', 'lseqmdl').
    method_full : str
        Full name of the method (e.g., 'Linear Scaling', 'LSEQM').
    folder : str
        Directory where the corrected precipitation will be saved.
    dekad_str : str
        String representing the dekad (e.g., '01', '11', '21').
    month_str : str
        Two-digit string representing the month (e.g., '01', '02', ..., '12').

    Returns:
    -------
    str or None
        Path to the saved file, or None if saving failed or skipped.
    """
    # Generate output filename
    from . import config as _cfg
    output_file = output_filename_template.format(
        folder=folder,
        filename_prefix=_cfg.FILENAME_PREFIX,
        method_abbr=method_abbr,
        month_str=month_str,
        dekad_str=dekad_str
    )

    # Check if output file exists
    if os.path.exists(output_file):
        logging.info(f"File {output_file} already exists.")
        decision = set_user_decision()

        if decision == 'S':
            logging.info(f"Skipping file {output_file}")
            # File is there on disk - return its path so callers can find it
            # for downstream inspection / plotting steps.
            return output_file
        elif decision == 'A':
            logging.info("Aborting process.")
            raise BiasCorrectAbort("User chose to abort the bias correction process.")
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
            'Conventions': 'CF-1.8',
            'cdm_data_type': 'GRID',
            'title': f'Bias Corrected IMERG Late Precipitation using {method_full}',
            'summary': f'Precipitation data corrected using {method_full}',
            'source': 'IMERG and CPC-UNI',
            'history': f'Created on {pd.Timestamp.now()}',
            'references': 'https://doi.org/10.3390/rs18142298',
            'DOI': '10.5067/GPM/IMERGDL/DAY/07',
            'creator_name': 'Benny Istanto',
            'creator_role': 'Climate Geographer',
            'creator_email': 'bennyistanto@apps.ipb.ac.id',
            'comment': f'This dataset has been bias corrected using {method_full}',
            **_run_provenance(),
        }
    )

    # Update metadata attributes.
    # No standard_name is set. The previous value 'corrected_precipitation' is
    # not in the CF standard name table, and CF treats standard_name as
    # optional: omitting it is correct when no controlled term applies, and is
    # preferable to asserting one that may not match. long_name carries the
    # description, and the correction method is in the filename and title.
    # units follow the IMERG-L source ('mm/day'), which is what this product is
    # derived from and what long_name describes. CPC-UNI labels the same
    # quantity 'mm' as a daily total; the values are identical either way.
    corrected_ds['precipitation'].attrs.update({
        'units': 'mm/day',
        'long_name': 'Corrected daily mean precipitation rate estimate'
    })

    corrected_ds['lat'].attrs.update({
        'units': 'degrees_north', 'long_name': 'Latitude', 'standard_name': 'latitude'
    })
    corrected_ds['lon'].attrs.update({
        'units': 'degrees_east', 'long_name': 'Longitude', 'standard_name': 'longitude'
    })

    # Apply land-sea mask to the `precipitation` variable only.
    # Use dynamic config lookup so a mid-session initialize_config() switch
    # (e.g. config.yml -> config_bali.yml) picks up the new mask path.
    masked_precip = apply_land_sea_mask(corrected_ds['precipitation'], _cfg.mask_file)

    # Replace the precipitation variable in corrected_ds with the masked version:
    corrected_ds['precipitation'] = masked_precip

    # Save to NetCDF following CF Convention
    try:
        corrected_ds.to_netcdf(output_file, encoding=cf18_f32, engine=NETCDF_ENGINE)
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

    Parameters:
    ----------
    ds : xarray.Dataset
        Dataset with a 'time' dimension.
    month : int
        Month number (1-12).

    Returns:
    ----------
    int
        Maximum number of days in the specified month across all years in the dataset.
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
        dekad_end_day,
        imerg_var=None,
        cpc_var=None
    ):
    """
    Aggregate IMERG and CPC data across all years for the specified dekad.

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
    imerg_var : str, optional
        Variable name for precipitation in IMERG dataset. If None, uses config default.
    cpc_var : str, optional
        Variable name for precipitation in CPC dataset. If None, uses config default.

    Returns:
    ----------
    tuple of xarray.DataArray
        Aggregated IMERG and CPC data for the specified dekad across all years.
    """
    # Use config defaults if variable names not provided
    if imerg_var is None:
        imerg_var = IMERG_PRECIP_VAR
    if cpc_var is None:
        cpc_var = CPC_PRECIP_VAR

    # Align datasets on their shared coordinates (time, lat, lon).
    # Uses join="inner" so only coordinates present in BOTH datasets are kept.
    # IMPORTANT: The caller must ensure the two datasets are already on the
    # same spatial grid (e.g., via reindex_and_align_with_monotonicity).
    # If IMERG and CPC have different native grids, the inner join will
    # find zero overlapping lat/lon values and produce empty arrays.
    logging.info("Aligning IMERG and CPC datasets...")
    n_lat_imerg_before = len(imerg_ds.lat)
    n_lat_cpc_before = len(cpc_ds.lat)

    imerg_ds, cpc_ds = xr.align(imerg_ds, cpc_ds, join="inner")

    # Diagnostic: detect spatial coordinate mismatch early
    n_lat_after = len(imerg_ds.lat)
    n_lon_after = len(imerg_ds.lon)
    if n_lat_after == 0 or n_lon_after == 0:
        logging.error(
            "Spatial coordinate mismatch detected! "
            f"IMERG had {n_lat_imerg_before} lats, CPC had {n_lat_cpc_before} lats, "
            f"but inner join produced {n_lat_after} lats and {n_lon_after} lons. "
            "This usually means the CPC dataset was not spatially aligned to IMERG "
            "before calling this function. Use reindex_and_align_with_monotonicity() "
            "first, then pass the aligned CPC dataset here."
        )
        raise ValueError(
            "No overlapping spatial coordinates between IMERG and CPC. "
            "Did you pass cpc_ds_aligned (from reindex_and_align_with_monotonicity) "
            "instead of the original cpc_ds?"
        )

    # Log time step counts after alignment
    logging.info(f"After alignment: IMERG has {len(imerg_ds.time)} time steps, CPC has {len(cpc_ds.time)} time steps")

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
        imerg_dekad_data = imerg_ds[imerg_var].where(imerg_time_mask, drop=True)
        cpc_dekad_data = cpc_ds[cpc_var].where(cpc_time_mask, drop=True)
    except KeyError as e:
        logging.error(f"Variable not found in dataset: {e}")
        logging.info(f"IMERG variables: {list(imerg_ds.data_vars)}")
        logging.info(f"CPC variables: {list(cpc_ds.data_vars)}")
        raise ValueError(f"Variable not found. Check IMERG_PRECIP_VAR='{imerg_var}' and CPC_PRECIP_VAR='{cpc_var}' in config.")
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


# ----
# Aggregate CPC at native 0.5° resolution for one dekad across all years
def aggregate_cpc_native_for_dekad(
        cpc_ds,
        month,
        dekad_start_day,
        dekad_end_day,
        cpc_var=None
    ):
    """
    Aggregate CPC data at its native 0.5° resolution for a specified dekad
    across all years. Unlike aggregate_data_across_years(), this function does
    NOT align to the IMERG grid - it preserves the CPC native coordinates.

    This is used by the native-resolution CPC parameter fitting (Option B)
    to avoid the 5×5 block artefact introduced by nearest-neighbour regridding.

    Parameters
    ----------
    cpc_ds : xarray.Dataset
        CPC precipitation dataset at native ~0.5° resolution.
    month : int
        Month number (1–12).
    dekad_start_day : int
        Start day of the dekad (1, 11, or 21).
    dekad_end_day : int
        End day of the dekad (10, 20, or last day of month).
    cpc_var : str, optional
        Variable name for precipitation in CPC dataset. If None, uses config default.

    Returns
    -------
    xarray.DataArray
        CPC precipitation for the dekad across all years at native resolution.
        Shape ~ (n_time, n_lat_cpc, n_lon_cpc).
    """
    if cpc_var is None:
        cpc_var = CPC_PRECIP_VAR

    # Ensure latitude is ascending (CPC sometimes has descending lat)
    if len(cpc_ds.lat) > 1 and cpc_ds.lat.values[0] > cpc_ds.lat.values[-1]:
        logging.info("CPC native lat is descending - flipping to ascending.")
        cpc_ds = cpc_ds.reindex(lat=cpc_ds.lat[::-1])

    # Create time mask for the specified dekad
    cpc_time_mask = (
        (cpc_ds['time.month'] == month) &
        (cpc_ds['time.day'] >= dekad_start_day) &
        (cpc_ds['time.day'] <= dekad_end_day)
    )

    # Apply time mask
    try:
        cpc_dekad_data = cpc_ds[cpc_var].where(cpc_time_mask, drop=True)
    except KeyError as e:
        logging.error(f"CPC variable not found: {e}")
        logging.info(f"CPC variables: {list(cpc_ds.data_vars)}")
        raise ValueError(f"Variable not found. Check CPC_PRECIP_VAR='{cpc_var}' in config.")

    # Validate
    if cpc_dekad_data.size == 0:
        logging.error("No CPC data available after masking.")
        raise ValueError("No CPC data found for the specified month and dekad.")

    logging.info(
        f"CPC native dekad data: {cpc_dekad_data.shape} "
        f"(lat {cpc_dekad_data.lat.values[0]:.2f}–{cpc_dekad_data.lat.values[-1]:.2f}, "
        f"lon {cpc_dekad_data.lon.values[0]:.2f}–{cpc_dekad_data.lon.values[-1]:.2f})"
    )

    return cpc_dekad_data
