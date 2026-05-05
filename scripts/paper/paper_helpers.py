"""
paper_helpers — common loaders, paths, constants for paper-only analyses.

This module provides path resolution and dataset loaders that match the
existing notebook 07 conventions. It is the only place that imports from
`src/`; downstream paper modules should import from here, not from `src/`
directly, so that the paper layer can be reorganized independently of the
package.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr

# --- Locate project root regardless of CWD ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Initialize src config from the canonical config.yml so all paths resolve.
from src.config import initialize_config
initialize_config(os.path.join(PROJECT_ROOT, 'config.yml'))
from src import config  # noqa: E402

# --- Constants matching nb07 ---
METHODS = ['ls', 'lseqm', 'lseqmdl']
METHOD_LABELS = ['LS', 'LSEQM', 'LSEQM+DL']
DEKAD_MAP = {1: '01', 2: '11', 3: '21'}
ALL_PERIODS = [(m, d) for m in range(1, 13) for d in [1, 2, 3]]

OUTPUT_DIR = config.output_dir


# --- Path helpers (match nb07 cell 3) ---
def metrics_path(method, month, dekad):
    """Path to single-dekad metrics NetCDF for cpc-vs-corrected."""
    dd = DEKAD_MAP[dekad]
    fname = (f'{config.FILENAME_PREFIX}_metricssd_cpc_imergl_'
             f'{method}_month{month:02d}_dekad{dd}.nc4')
    return os.path.join(OUTPUT_DIR, f'metrics_{method}', fname)


def station_val_path(method, month, dekad):
    """Path to per-station validation CSV for the given method/period."""
    dd = DEKAD_MAP[dekad]
    fname = f'station_validation_{method}_month{month:02d}_dekad{dd}.csv'
    return os.path.join(OUTPUT_DIR, 'station_validation', fname)


def corrected_file(method, month, dekad):
    """Path to corrected precipitation NetCDF for given method/period."""
    dd = DEKAD_MAP[dekad]
    folder_attr = {
        'ls': config.ls_corrected_precip_path,
        'lseqm': config.lseqm_corrected_precip_path,
        'lseqmdl': config.lseqmdl_corrected_precip_path,
    }[method]
    fname = (f'{config.FILENAME_PREFIX}_{method}_corrected_imergl_'
             f'month{month:02d}_dekad{dd}.nc4')
    return os.path.join(folder_attr, fname)


# --- Dataset loaders ---
def load_corrected_da(method, month, dekad):
    """Load a corrected precipitation DataArray for one method/period."""
    fpath = corrected_file(method, month, dekad)
    if not os.path.exists(fpath):
        return None
    ds = xr.open_dataset(fpath, decode_times=True)
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    return ds[config.IMERG_PRECIP_VAR]


def load_reference_da(label):
    """Load CPC, IMERG-L, or IMERG-F as a DataArray (full domain, full time)."""
    label = label.lower()
    if label == 'cpc':
        ds = xr.open_dataset(config.cpc_file, decode_times=True)
        var = config.CPC_PRECIP_VAR
    elif label == 'imergl':
        ds = xr.open_dataset(config.imergl_file, decode_times=True)
        var = config.IMERG_PRECIP_VAR
    elif label == 'imergf':
        ds = xr.open_dataset(config.imergf_file, decode_times=True)
        var = config.IMERG_PRECIP_VAR
    else:
        raise ValueError(f"Unknown reference label: {label}")
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    return ds[var]


def load_station_obs():
    """Load BMKG station observations as a DataFrame indexed by date."""
    from src.station_validation import load_station_observations
    return load_station_observations(
        config.STATION_DATA_FILE,
        station_location_file=config.STATION_FILE,
    )


def load_station_locations_df():
    """Load BMKG station locations as a DataFrame."""
    from src.station_density import load_station_locations
    return load_station_locations(config.STATION_FILE)


# --- Period helpers ---
def dekad_label(month, dekad_idx):
    """Return short label for a (month, dekad_idx) tuple."""
    return f"M{month:02d}D{DEKAD_MAP[dekad_idx]}"


def dekad_index_for_date(date):
    """Return (month, dekad_idx) for a given pandas Timestamp."""
    m = int(date.month)
    d = int(date.day)
    if d <= 10:
        return m, 1
    elif d <= 20:
        return m, 2
    else:
        return m, 3
