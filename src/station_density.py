"""
Module: station_density.py

Computes a station density confidence mask from gauge station locations.
The confidence mask quantifies how reliable CPC-UNI is at each grid cell,
based on the number of gauge stations that contribute to the CPC interpolation
in that area.

The mask is used to spatially modulate the DL blending alpha in
apply_deeplearning_model():
  - High confidence (many stations): DL gets more influence (trust CPC target)
  - Low confidence (no stations): revert to pure LSEQM (trust physics)

The confidence value at each pixel controls the effective blending alpha via:
    effective_alpha = 1.0 - confidence * (1.0 - base_alpha)

This means:
    confidence = 1.0  =>  effective_alpha = base_alpha (e.g., 0.7)
    confidence = 0.0  =>  effective_alpha = 1.0 (pure LSEQM, DL disabled)

The module follows the same caching and I/O patterns as utility.py (load_mask)
and io.py (save_corrected_precip).

**Author**:
  Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from .config import (DENSITY_CPC_RESOLUTION,
                     DENSITY_SMOOTHING_SIGMA,
                     DENSITY_SATURATION_COUNT,
                     DENSITY_LAT_RANGE,
                     DENSITY_LON_RANGE)

# +++++++++++++++++++++++++++++++++++++++++
# Module-level cache (same pattern as _mask_cache in utility.py)
# +++++++++++++++++++++++++++++++++++++++++
_confidence_cache = {}


# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# ----
# Load station locations from CSV
def load_station_locations(station_file):
    """
    Load gauge station locations from a CSV file.

    Parameters
    ----------
    station_file : str
        Path to CSV file with columns including 'Lon' and 'Lat'.
        Expected format: ID, ID_WMO, Station, Lon, Lat, Elevation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with station data. Must contain 'Lon' and 'Lat' columns.

    Raises
    ------
    FileNotFoundError
        If the station file does not exist.
    ValueError
        If required columns ('Lon', 'Lat') are missing.
    """
    if not os.path.isfile(station_file):
        raise FileNotFoundError(f"Station file not found: {station_file}")

    df = pd.read_csv(station_file, sep=None, engine='python')

    # Validate required columns
    for col in ('Lon', 'Lat'):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in station file. "
                f"Available columns: {list(df.columns)}"
            )

    # Drop rows with missing coordinates
    n_before = len(df)
    df = df.dropna(subset=['Lon', 'Lat'])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logging.warning(f"Dropped {n_dropped} stations with missing coordinates")

    logging.info(f"Loaded {len(df)} station locations from {station_file}")
    return df


# ----
# Count stations per CPC native cell
def count_stations_per_cell(station_df,
                            cpc_resolution=DENSITY_CPC_RESOLUTION,
                            lat_range=DENSITY_LAT_RANGE,
                            lon_range=DENSITY_LON_RANGE):
    """
    Count the number of gauge stations in each CPC native-resolution cell.

    CPC-UNI operates at 0.5 degree resolution. The quality of CPC-UNI at
    any location depends primarily on how many gauge stations contribute
    to that 0.5-degree cell. This function bins station coordinates into
    a regular grid at CPC's native resolution.

    Parameters
    ----------
    station_df : pandas.DataFrame
        Station locations with 'Lon' and 'Lat' columns.
    cpc_resolution : float, optional
        CPC native grid spacing in degrees (default 0.5).
    lat_range : tuple of float, optional
        (south, north) bounds for the counting grid.
    lon_range : tuple of float, optional
        (west, east) bounds for the counting grid.

    Returns
    -------
    xarray.DataArray
        2-D array of station counts on the CPC native grid,
        with 'lat' and 'lon' coordinates at cell centers.
    """
    # Build bin edges (half-cell offset so cell centers align with CPC grid)
    lat_edges = np.arange(lat_range[0], lat_range[1] + cpc_resolution, cpc_resolution)
    lon_edges = np.arange(lon_range[0], lon_range[1] + cpc_resolution, cpc_resolution)

    # Cell centers
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    # Count stations per cell using 2D histogram
    station_lats = station_df['Lat'].values
    station_lons = station_df['Lon'].values

    counts, _, _ = np.histogram2d(
        station_lats, station_lons,
        bins=[lat_edges, lon_edges]
    )

    # Log station distribution summary
    n_cells_with_stations = np.sum(counts > 0)
    n_total_cells = counts.size
    logging.info(
        f"Station density: {int(counts.sum())} stations in "
        f"{n_cells_with_stations}/{n_total_cells} cells "
        f"(CPC {cpc_resolution}° grid, "
        f"max {int(counts.max())} stations/cell)"
    )

    # Convert to xarray DataArray
    count_da = xr.DataArray(
        counts,
        dims=('lat', 'lon'),
        coords={'lat': lat_centers, 'lon': lon_centers},
        attrs={
            'long_name': 'Number of gauge stations per CPC grid cell',
            'units': '1',
            'grid_resolution_degrees': cpc_resolution,
        }
    )

    return count_da


# ----
# Convert station counts to confidence map
def compute_confidence_map(station_counts,
                           smoothing_sigma=DENSITY_SMOOTHING_SIGMA,
                           saturation_count=DENSITY_SATURATION_COUNT):
    """
    Convert raw station counts to a [0, 1] confidence map.

    Steps:
    1. Apply Gaussian smoothing to the count grid to prevent hard cell-boundary
       artifacts. With sigma=1.0 at 0.5° resolution, the e-folding distance
       is approximately 55 km — one neighboring cell in each direction.
    2. Normalize: confidence = min(count_smooth / saturation_count, 1.0).
       A saturation_count of 3 means that 3 or more (smoothed) stations
       produce full confidence (1.0).

    Parameters
    ----------
    station_counts : xarray.DataArray
        Raw station counts on the CPC native grid.
    smoothing_sigma : float, optional
        Gaussian filter sigma in grid-cell units (default 1.0).
    saturation_count : float, optional
        Number of (smoothed) stations for full confidence (default 3).

    Returns
    -------
    xarray.DataArray
        Confidence map on the CPC native grid, values in [0, 1].
    """
    # Apply Gaussian smoothing
    counts_smooth = gaussian_filter(
        station_counts.values.astype(float),
        sigma=smoothing_sigma
    )

    # Normalize to [0, 1]
    confidence = np.clip(counts_smooth / saturation_count, 0.0, 1.0)

    # Log confidence distribution
    logging.info(
        f"Confidence map: min={confidence.min():.3f}, "
        f"max={confidence.max():.3f}, "
        f"mean={confidence.mean():.3f} "
        f"(sigma={smoothing_sigma}, saturation={saturation_count})"
    )

    confidence_da = xr.DataArray(
        confidence,
        dims=station_counts.dims,
        coords=station_counts.coords,
        attrs={
            'long_name': 'CPC station density confidence',
            'units': '1',
            'valid_range': [0.0, 1.0],
            'smoothing_sigma_gridcells': smoothing_sigma,
            'saturation_count': saturation_count,
            'comment': (
                'Confidence = 1.0 where CPC-UNI has many gauge stations nearby; '
                '0.0 where no stations contribute. Used to spatially modulate '
                'the DL blending alpha: effective_alpha = 1 - confidence * (1 - base_alpha).'
            ),
        }
    )

    return confidence_da


# ----
# Upscale confidence to working grid
def upscale_confidence_to_working_grid(confidence_coarse, target_lat, target_lon):
    """
    Upscale the CPC-resolution confidence map to the 0.1-degree working grid
    using nearest-neighbor interpolation.

    This follows the same interpolation approach used for the land-sea mask
    in utility.apply_land_sea_mask().

    Parameters
    ----------
    confidence_coarse : xarray.DataArray
        Confidence map on CPC native grid (0.5 deg).
    target_lat : array-like
        Target latitude coordinates (from the IMERG/working grid).
    target_lon : array-like
        Target longitude coordinates (from the IMERG/working grid).

    Returns
    -------
    xarray.DataArray
        Confidence map interpolated to the working grid, shape (target_lat, target_lon).
    """
    confidence_fine = confidence_coarse.interp(
        lat=target_lat, lon=target_lon, method='nearest'
    )

    # Fill any edge NaN values (stations near grid boundary) with 0 (no confidence)
    confidence_fine = confidence_fine.fillna(0.0)

    logging.info(
        f"Upscaled confidence mask: {confidence_coarse.shape} -> {confidence_fine.shape}"
    )

    return confidence_fine


# ----
# End-to-end pipeline
def build_confidence_mask(station_file, target_lat, target_lon,
                          cpc_resolution=DENSITY_CPC_RESOLUTION,
                          smoothing_sigma=DENSITY_SMOOTHING_SIGMA,
                          saturation_count=DENSITY_SATURATION_COUNT,
                          lat_range=DENSITY_LAT_RANGE,
                          lon_range=DENSITY_LON_RANGE):
    """
    End-to-end pipeline: CSV -> confidence mask on the working grid.

    Chains:
      load_station_locations -> count_stations_per_cell ->
      compute_confidence_map -> upscale_confidence_to_working_grid.

    Parameters
    ----------
    station_file : str
        Path to station CSV.
    target_lat, target_lon : array-like
        Working grid coordinates (from the IMERG dataset).
    cpc_resolution : float, optional
        CPC native resolution in degrees (default 0.5).
    smoothing_sigma : float, optional
        Gaussian smoothing sigma in grid-cell units (default 1.0).
    saturation_count : float, optional
        Number of (smoothed) stations for full confidence (default 3).
    lat_range, lon_range : tuple of float, optional
        Grid bounds for station counting.

    Returns
    -------
    xarray.DataArray
        Confidence mask on the working grid, values in [0, 1].
    """
    logging.info("Building station density confidence mask...")

    # Step 1: Load station locations
    station_df = load_station_locations(station_file)

    # Step 2: Count stations per CPC cell
    station_counts = count_stations_per_cell(
        station_df, cpc_resolution=cpc_resolution,
        lat_range=lat_range, lon_range=lon_range
    )

    # Step 3: Smooth and normalize to confidence [0, 1]
    confidence_coarse = compute_confidence_map(
        station_counts,
        smoothing_sigma=smoothing_sigma,
        saturation_count=saturation_count
    )

    # Step 4: Upscale to working grid
    confidence_fine = upscale_confidence_to_working_grid(
        confidence_coarse, target_lat, target_lon
    )

    logging.info("Station density confidence mask built successfully.")

    return confidence_fine


# ----
# Save confidence mask to NetCDF
def save_confidence_mask(confidence_mask, output_file):
    """
    Save the confidence mask to a NetCDF file with CF-1.8 compliant metadata.

    Parameters
    ----------
    confidence_mask : xarray.DataArray
        Confidence mask to save (2-D, lat × lon, values in [0, 1]).
    output_file : str
        Output file path (.nc4).
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Build xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'confidence': (
                ('lat', 'lon'),
                confidence_mask.values,
                confidence_mask.attrs
            )
        },
        coords={
            'lat': confidence_mask.lat.values,
            'lon': confidence_mask.lon.values,
        },
        attrs={
            'Conventions': 'CF-1.8',
            'title': 'CPC-UNI Station Density Confidence Mask',
            'institution': 'Bogor Agricultural University / The World Bank',
            'source': 'BMKG weather station locations',
            'comment': (
                'Confidence mask based on gauge station density at CPC-UNI '
                'native resolution (0.5 deg). Values range from 0 (no nearby '
                'stations, CPC unreliable) to 1 (many stations, CPC reliable). '
                'Used to spatially modulate the DL blending alpha.'
            ),
        }
    )

    # CF-1.8 encoding
    from .config import NETCDF_ENGINE
    encoding = {
        'confidence': {
            'dtype': 'float32',
            'zlib': True,
            '_FillValue': np.nan,
        }
    }

    ds.to_netcdf(output_file, encoding=encoding, engine=NETCDF_ENGINE)
    logging.info(f"Confidence mask saved to {output_file}")


# ----
# Load and cache confidence mask
def load_confidence_mask(mask_file):
    """
    Load a previously saved confidence mask from NetCDF, with caching.

    Follows the same caching pattern as utility.load_mask(): the mask is
    loaded once per unique file path and cached in memory for subsequent
    calls.

    Parameters
    ----------
    mask_file : str
        Path to the confidence mask NetCDF file.

    Returns
    -------
    xarray.DataArray
        Cached confidence mask (2-D, lat × lon, values in [0, 1]).

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.
    """
    if mask_file in _confidence_cache:
        logging.debug(f"Confidence mask loaded from cache: {mask_file}")
        return _confidence_cache[mask_file]

    if not os.path.isfile(mask_file):
        raise FileNotFoundError(f"Confidence mask file not found: {mask_file}")

    from .config import NETCDF_ENGINE
    ds = xr.open_dataset(mask_file, engine=NETCDF_ENGINE)
    confidence = ds['confidence'].load()  # Load into memory so file handle can close
    ds.close()

    _confidence_cache[mask_file] = confidence

    logging.info(
        f"Loaded confidence mask from {mask_file} "
        f"(shape: {confidence.shape}, range: [{float(confidence.min()):.3f}, "
        f"{float(confidence.max()):.3f}])"
    )

    return confidence


# ----
# Main entry point: get or create
def get_or_create_confidence_mask(station_file, confidence_mask_file,
                                  target_lat, target_lon, **kwargs):
    """
    Load the confidence mask from file if it exists, otherwise compute and save.

    This is the main entry point for notebook use. On the first run, it builds
    the confidence mask from the station CSV and saves it as NetCDF. On
    subsequent runs, it loads the pre-computed mask from file.

    Parameters
    ----------
    station_file : str
        Path to the station CSV file.
    confidence_mask_file : str
        Path where the confidence mask NetCDF should be saved/loaded.
    target_lat, target_lon : array-like
        Working grid coordinates (from the IMERG/LSEQM dataset).
    **kwargs
        Additional keyword arguments passed to build_confidence_mask()
        (cpc_resolution, smoothing_sigma, saturation_count, lat_range,
        lon_range).

    Returns
    -------
    xarray.DataArray
        Confidence mask on the working grid, values in [0, 1].
    """
    if os.path.isfile(confidence_mask_file):
        logging.info(f"Loading existing confidence mask: {confidence_mask_file}")
        mask = load_confidence_mask(confidence_mask_file)

        # Validate that the loaded mask matches the target grid dimensions
        if (mask.shape[0] != len(target_lat) or
                mask.shape[1] != len(target_lon)):
            logging.warning(
                f"Cached confidence mask shape {mask.shape} does not match "
                f"target grid ({len(target_lat)}, {len(target_lon)}). "
                f"Rebuilding..."
            )
        else:
            return mask

    # Build from scratch
    logging.info("Confidence mask not found, building from station data...")
    mask = build_confidence_mask(
        station_file, target_lat, target_lon, **kwargs
    )

    # Save for future runs
    save_confidence_mask(mask, confidence_mask_file)

    return mask
