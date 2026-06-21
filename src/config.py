"""
Module: config.py

This module defines the configuration for the Hybrid Bias Correction (LSEQM+DL) workflow.
It loads settings from a YAML configuration file and provides them as module-level variables.

Configuration is loaded via initialize_config(), which should be called once at startup
(typically in the notebook or main script). If not called, default values are used.

The configuration includes:
- Directory settings for the main project, input data, and output locations.
- Default input file paths for IMERG, CPC, and the land-sea mask.
- Variable names for precipitation in each dataset.
- Output directories and filename templates.
- CF-compliant encoding settings for NetCDF outputs.
- Statistical fitting and quantile mapping parameters (GPD, EQM).
- Deep learning model training parameters.
- Runtime settings (interactive mode, default actions for existing files).

**Author**:
  Benny Istanto
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""
import os
import sys
import logging
import numpy as np

# +++++++++++++++++++++++++++++++++++++++++
# Logging Setup
# +++++++++++++++++++++++++++++++++++++++++

def setup_logging(level=logging.INFO):
    """
    Configure logging so messages appear in Colab, Jupyter, and scripts.

    Google Colab's IPython kernel installs its own root-logger handler
    before user code runs, which silently swallows ``logging.basicConfig()``
    calls (because ``basicConfig`` is a no-op when handlers already exist).

    This function works around that by:

    * Using ``force=True`` (Python 3.8+) to replace any pre-existing config.
    * Attaching a ``StreamHandler(sys.stdout)`` so output lands in the
      notebook cell, not in a hidden stderr stream.

    Safe to call multiple times - subsequent calls just reset the level.

    Parameters
    ----------
    level : int, optional
        Logging level (default ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# Auto-configure on import so every ``logging.info()`` in src/ is visible
setup_logging()


# +++++++++++++++++++++++++++++++++++++++++
# Default Configuration Values
# +++++++++++++++++++++++++++++++++++++++++

# These defaults work if no config.yml is loaded.
# They can be overridden by calling initialize_config() with a YAML file path.

# --- General ---
FILENAME_PREFIX = 'idn_cli'             # Country/project prefix for all output filenames

# --- Area of Interest ---
AOI_LAT_RANGE = (-11.0, 6.0)            # Bounding box (south, north)
AOI_LON_RANGE = (95.0, 141.0)           # Bounding box (west, east)

# --- Directories ---
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
input_dir = os.path.join(main_dir, 'data', 'input')
output_dir = os.path.join(main_dir, 'data', 'output')

# --- Input Files ---
imergl_file = os.path.join(input_dir, 'imergl', 'idn_imergl.nc4')
imergf_file = os.path.join(input_dir, 'imergf', 'idn_imergf.nc4')
cpc_file = os.path.join(input_dir, 'cpcuni', 'idn_cpcuni.nc4')
cpc_native_file = os.path.join(input_dir, 'cpcuni', 'idn_cpcuni_native05.nc4')
mask_file = os.path.join(main_dir, 'data', 'subset', 'iso3', 'idn_subset.nc')

# --- Variable Names ---
IMERG_PRECIP_VAR = 'precipitation'
CPC_PRECIP_VAR = 'precip'
MASK_VAR = 'land'

# --- Output Directories ---
ls_corrected_precip_path = os.path.join(output_dir, 'corrected_ls')
lseqm_corrected_precip_path = os.path.join(output_dir, 'corrected_lseqm')
lseqmdl_corrected_precip_path = os.path.join(output_dir, 'corrected_lseqmdl')
trained_models_path = os.path.join(output_dir, 'trained_models')
metrics_path_template = os.path.join(output_dir, 'metrics_{method}')
quality_path_template = os.path.join(output_dir, 'quality_{method}')

# --- Output Filename Template ---
output_filename_template = "{folder}/{filename_prefix}_{method_abbr}_corrected_imergl_month{month_str}_dekad{dekad_str}.nc4"

# --- NetCDF Encoding (CF 1.8) ---
cf18_f32 = {'precipitation': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan}}

# --- Statistical Parameters ---
N_SPLITS_GPD_CROSSVALIDATE = 5
GPD_THRESHOLD_PERCENTILE = 80
UPPER_CAP_THRESHOLD_PERCENTILE = 99.9

# --- Performance Metrics Parameters ---
WET_DAY_THRESHOLD = 1.0  # mm/day threshold for categorical metrics

# --- Quality Assessment Weights ---
QA_COMPONENT_WEIGHTS = {'basic_stats': 0.4, 'distribution': 0.3, 'temporal': 0.3}
QA_CATEGORICAL_THRESHOLDS = {'excellent': 0.8, 'good': 0.6, 'fair': 0.4}

# --- Visualisation / Admin Boundaries ---
# Region-agnostic: if BOUNDARIES is empty or no files resolve, src.visualisation
# falls back to cartopy Natural Earth so the framework stays usable in any AOI.
BOUNDARY_DIR = None
BOUNDARIES = []

# Padding (in degrees) added to each side of the AOI bbox when drawing map
# axes, so coastline / boundary outlines have breathing room at the edges.
# 0.0 = AOI flush with panel edge; 0.1 deg (default) = one IMERG cell of margin.
MAP_EXTENT_PAD = 0.1

# --- Deep Learning Parameters ---
DL_EPOCHS = 50
DL_BATCH_SIZE = 64
DL_VALIDATION_SPLIT = 0.2
DL_EARLY_STOPPING_PATIENCE = 5
DL_DROPOUT_RATE_1 = 0.2
DL_DROPOUT_RATE_2 = 0.3
DL_DROPOUT_RATE_DENSE = 0.4
DL_FILTER_SIZE_1 = (3, 3)
DL_FILTER_SIZE_2 = (3, 3)
DL_NUM_FILTERS_1 = 32
DL_NUM_FILTERS_2 = 64
DL_DENSE_LAYER_SIZE = 128
DL_OPTIMIZER = 'adam'
DL_RANDOM_SEED = 0         # Seed for weight init, validation-split shuffle, and dropout masks.
                           # Set to None in config.yml (deep_learning.random_seed: null) to
                           # restore the original stochastic behaviour.
DL_BLEND_ALPHA = 0.7       # Weight for LSEQM in the DL blending step.
                           # final = alpha * LSEQM + (1 - alpha) * DL_predicted
                           # 1.0 = pure LSEQM (DL disabled), 0.0 = pure DL.
                           # Recommended range: 0.5–0.8.  Higher values trust
                           # the physical-statistical result more; lower values
                           # give DL more influence on extreme pixels.

# --- Station Density Confidence Mask ---
# Spatially modulates blend_alpha based on CPC-UNI gauge station density.
# Set USE_CONFIDENCE_MASK to True AND provide STATION_FILE to enable.
USE_CONFIDENCE_MASK = False          # Master toggle; False = uniform blend_alpha
STATION_FILE = None                  # Path to station CSV; None = disabled
CONFIDENCE_MASK_FILE = None          # Path to pre-computed confidence mask
DENSITY_CPC_RESOLUTION = 0.5        # CPC native grid spacing (degrees)
DENSITY_SMOOTHING_SIGMA = 1.0       # Gaussian smoothing (grid-cell units)
DENSITY_SATURATION_COUNT = 2        # Stations for full confidence (1.0)
DENSITY_LAT_RANGE = (-11.0, 6.0)    # Grid bounds (south, north)
DENSITY_LON_RANGE = (95.0, 141.0)   # Grid bounds (west, east)

# --- Station Validation ---
# Independent validation of bias-corrected products against station observations.
# Uses the same station location file as station_density (STATION_FILE above).
STATION_DATA_FILE = None             # Path to daily precipitation obs CSV
STATION_VALIDATION_OUTPUT_DIR = None # Set during initialize_config()
MISSING_SENTINEL = 8888.0            # Missing data sentinel in station CSV
MIN_VALID_DAYS = 30                  # Min paired days for reliable station metrics

# --- Region Mapping ---
# Province-to-region dictionary for spatial aggregation.
# Loaded from config.yml; empty dict means no regional grouping available.
REGION_MAPPING = {}
ISLAND_ORDER = []

# --- Runtime Settings ---
INTERACTIVE = True
EXISTING_FILE_ACTION = 'skip'       # 'overwrite', 'skip', or 'abort'
EXISTING_MODEL_ACTION = 'use_existing'  # 'use_existing', 'overwrite', or 'abort'

# --- NetCDF Engine ---
# Detected at import time. Prefer 'netcdf4'; fall back to 'h5netcdf' when the
# netCDF4 C library is unavailable (common on Windows conda-forge builds).
NETCDF_ENGINE = None  # Will be set by _detect_netcdf_engine()


def _detect_netcdf_engine():
    """
    Detect which xarray NetCDF engine is available.

    Tries ``netCDF4`` first (the most common engine). If that import fails - 
    for example because of a missing or broken DLL on Windows - falls back to
    ``h5netcdf``, which is a pure-Python HDF5 backend with no C-library
    dependency chain beyond h5py.

    Returns
    -------
    str
        ``'netcdf4'`` or ``'h5netcdf'``.
    """
    try:
        import netCDF4 as _nc4  # noqa: F401
        return 'netcdf4'
    except (ImportError, OSError) as exc:
        logging.info(
            f"netCDF4 engine unavailable ({exc}); falling back to h5netcdf."
        )
        try:
            import h5netcdf  # noqa: F401
            return 'h5netcdf'
        except ImportError:
            logging.warning(
                "Neither netCDF4 nor h5netcdf is available. "
                "Install one of them: pip install netCDF4  OR  pip install h5netcdf"
            )
            return 'netcdf4'  # Return default; xarray will raise a clear error later


# Detect engine at module load time
NETCDF_ENGINE = _detect_netcdf_engine()


# +++++++++++++++++++++++++++++++++++++++++
# Portable Project Root Detection
# +++++++++++++++++++++++++++++++++++++++++

def find_project_root(extra_candidates=None):
    """
    Locate the project root using a **strong fingerprint**.

    A directory is accepted as the project root only when it contains
    ALL of the following (prevents false positives from unrelated projects):

    * ``src/__init__.py``
    * ``src/config.py``
    * ``config.yml`` **or** ``config.yaml``

    Works across Google Colab, WSL, Linux, macOS, and Windows without
    any hardcoded paths.  The search order is:

    1. Directory containing *this* file (``src/config.py`` → parent).
    2. Current working directory and its parent.
    3. ``extra_candidates`` supplied by the caller (if any).
    4. Google Drive deep scan (Colab only, up to 4 levels).

    Parameters
    ----------
    extra_candidates : list of str, optional
        Additional directory paths to consider.

    Returns
    -------
    str
        Absolute path to the project root.
    """

    def _is_root(path):
        """Strong check: src/__init__.py + src/config.py + config.y(a)ml."""
        has_src = (
            os.path.isfile(os.path.join(path, 'src', '__init__.py'))
            and os.path.isfile(os.path.join(path, 'src', 'config.py'))
        )
        has_cfg = (
            os.path.isfile(os.path.join(path, 'config.yml'))
            or os.path.isfile(os.path.join(path, 'config.yaml'))
        )
        return has_src and has_cfg

    candidates = []

    # 1. This file lives in <project_root>/src/config.py
    candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 2. CWD and its parent (handles Colab where CWD = /content)
    candidates.append(os.getcwd())
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

    # 3. Caller-provided extras
    if extra_candidates:
        candidates.extend(extra_candidates)

    # 4. Check all candidates (fast path)
    for cand in candidates:
        cand = os.path.abspath(cand)
        if _is_root(cand):
            return cand

    # 5. Colab deep scan - walk Google Drive (up to 4 levels)
    drive_root = '/content/drive/MyDrive'
    if os.path.isdir(drive_root):
        logging.info("Scanning Google Drive for project root ...")
        for dirpath, dirnames, _filenames in os.walk(drive_root):
            depth = dirpath.replace(drive_root, '').count(os.sep)
            if depth > 4:
                dirnames.clear()
                continue
            if _is_root(dirpath):
                logging.info(f"Found project root on Google Drive: {dirpath}")
                return dirpath

    # Fallback
    fallback = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logging.warning("Could not auto-detect project root. Using: %s", fallback)
    return fallback


# +++++++++++++++++++++++++++++++++++++++++
# Configuration Initialization
# +++++++++++++++++++++++++++++++++++++++++

def initialize_config(config_path=None):
    """
    Load configuration from a YAML file and update module-level variables.

    This function should be called once at the start of your notebook or script.
    If config_path is None, it looks for config.yml (or config.yaml) in the
    project root directory.

    Parameters:
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If None, searches for config.yml
        then config.yaml in the project root (one level above src/).

    Returns:
    ----------
    dict
        The loaded configuration dictionary.
    """
    global FILENAME_PREFIX, AOI_LAT_RANGE, AOI_LON_RANGE
    global main_dir, input_dir, output_dir
    global imergl_file, imergf_file, cpc_file, cpc_native_file, mask_file
    global IMERG_PRECIP_VAR, CPC_PRECIP_VAR, MASK_VAR
    global ls_corrected_precip_path, lseqm_corrected_precip_path
    global lseqmdl_corrected_precip_path, trained_models_path
    global metrics_path_template, quality_path_template
    global output_filename_template, cf18_f32
    global N_SPLITS_GPD_CROSSVALIDATE, GPD_THRESHOLD_PERCENTILE, UPPER_CAP_THRESHOLD_PERCENTILE
    global WET_DAY_THRESHOLD, QA_COMPONENT_WEIGHTS, QA_CATEGORICAL_THRESHOLDS
    global DL_EPOCHS, DL_BATCH_SIZE, DL_VALIDATION_SPLIT, DL_EARLY_STOPPING_PATIENCE
    global DL_DROPOUT_RATE_1, DL_DROPOUT_RATE_2, DL_DROPOUT_RATE_DENSE
    global DL_FILTER_SIZE_1, DL_FILTER_SIZE_2
    global DL_NUM_FILTERS_1, DL_NUM_FILTERS_2, DL_DENSE_LAYER_SIZE, DL_OPTIMIZER
    global DL_RANDOM_SEED, DL_BLEND_ALPHA
    global USE_CONFIDENCE_MASK, STATION_FILE, CONFIDENCE_MASK_FILE
    global DENSITY_CPC_RESOLUTION, DENSITY_SMOOTHING_SIGMA, DENSITY_SATURATION_COUNT
    global DENSITY_LAT_RANGE, DENSITY_LON_RANGE
    global MISSING_SENTINEL, MIN_VALID_DAYS, REGION_MAPPING, ISLAND_ORDER
    global STATION_DATA_FILE, STATION_VALIDATION_OUTPUT_DIR
    global INTERACTIVE, EXISTING_FILE_ACTION, EXISTING_MODEL_ACTION
    global NETCDF_ENGINE
    global BOUNDARY_DIR, BOUNDARIES, MAP_EXTENT_PAD

    try:
        import yaml
    except ImportError:
        logging.warning("PyYAML not installed. Using default configuration. "
                       "Install with: pip install pyyaml")
        _ensure_output_directories()
        return {}

    # Find config file - prefer config.yml, fall back to config.yaml
    if config_path is None:
        project_root = find_project_root()
        for candidate in ('config.yml', 'config.yaml'):
            config_path = os.path.join(project_root, candidate)
            if os.path.exists(config_path):
                break

    if not os.path.exists(config_path):
        logging.warning(f"Config file not found at {config_path}. Using default configuration.")
        _ensure_output_directories()
        return {}

    # Load YAML
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    logging.info(f"Loaded configuration from {config_path}")

    # --- General settings ---
    general = cfg.get('general', {})
    FILENAME_PREFIX = general.get('filename_prefix', FILENAME_PREFIX)

    # --- Area of interest ---
    aoi = cfg.get('aoi', {})
    aoi_lat = aoi.get('lat_range')
    if aoi_lat:
        AOI_LAT_RANGE = tuple(aoi_lat)
    aoi_lon = aoi.get('lon_range')
    if aoi_lon:
        AOI_LON_RANGE = tuple(aoi_lon)

    # --- Resolve directories with template substitution ---
    # The auto-detected project root (where src/__init__.py lives) is the
    # definitive location of the code on *this* machine.  config.yml may
    # contain a main_dir that points to a *different* copy of the project
    # (e.g. an older snapshot on the same Google Drive, or a WSL path that
    # happens to exist but isn't the copy being executed).
    #
    # Rule: use the config.yml value ONLY when it matches the directory
    # that actually contains *this* src/ package.  Otherwise fall back to
    # the auto-detected root.
    auto_root = find_project_root()
    dirs = cfg.get('directories', {})
    cfg_main_dir = dirs.get('main_dir')

    if cfg_main_dir and cfg_main_dir != 'auto' and os.path.isdir(cfg_main_dir):
        # Extra safety: verify this is the *same* project, not a different copy
        if os.path.abspath(cfg_main_dir) == os.path.abspath(auto_root):
            main_dir = cfg_main_dir
        else:
            logging.info(
                f"Configured main_dir '{cfg_main_dir}' exists but differs from "
                f"the project being executed ({auto_root}). Using auto-detected root."
            )
            main_dir = auto_root
    else:
        # Either main_dir is "auto", missing, or the path doesn't exist
        # on this machine.  Auto-detect using find_project_root().
        main_dir = auto_root
        if cfg_main_dir and cfg_main_dir != 'auto':
            logging.info(
                f"Configured main_dir '{cfg_main_dir}' does not exist on this system. "
                f"Using auto-detected project root: {main_dir}"
            )

    # Resolve {main_dir} in other directory paths
    raw_input_dir = dirs.get('input_dir', os.path.join(main_dir, 'data', 'input'))
    input_dir = raw_input_dir.replace('{main_dir}', main_dir)

    raw_output_dir = dirs.get('output_dir', os.path.join(main_dir, 'data', 'output'))
    output_dir = raw_output_dir.replace('{main_dir}', main_dir)

    # --- Resolve input files ---
    files = cfg.get('input_files', {})
    raw_imerg = files.get('imergl_file', imergl_file)
    imergl_file = raw_imerg.replace('{input_dir}', input_dir).replace('{main_dir}', main_dir)

    raw_imergf = files.get('imergf_file', imergf_file)
    imergf_file = raw_imergf.replace('{input_dir}', input_dir).replace('{main_dir}', main_dir)

    raw_cpc = files.get('cpc_file', cpc_file)
    cpc_file = raw_cpc.replace('{input_dir}', input_dir).replace('{main_dir}', main_dir)

    raw_cpc_native = files.get('cpc_native_file', cpc_native_file)
    cpc_native_file = raw_cpc_native.replace('{input_dir}', input_dir).replace('{main_dir}', main_dir)

    raw_mask = files.get('mask_file', mask_file)
    mask_file = raw_mask.replace('{input_dir}', input_dir).replace('{main_dir}', main_dir)

    # --- Variable names ---
    var_names = cfg.get('variable_names', {})
    IMERG_PRECIP_VAR = var_names.get('imerg_precip_var', IMERG_PRECIP_VAR)
    CPC_PRECIP_VAR = var_names.get('cpc_precip_var', CPC_PRECIP_VAR)
    MASK_VAR = var_names.get('mask_var', MASK_VAR)

    # --- Output directories ---
    out_dirs = cfg.get('output_dirs', {})
    raw_ls = out_dirs.get('ls_corrected', os.path.join(output_dir, 'corrected_ls'))
    ls_corrected_precip_path = raw_ls.replace('{output_dir}', output_dir)

    raw_lseqm = out_dirs.get('lseqm_corrected', os.path.join(output_dir, 'corrected_lseqm'))
    lseqm_corrected_precip_path = raw_lseqm.replace('{output_dir}', output_dir)

    raw_lseqmdl = out_dirs.get('lseqmdl_corrected', os.path.join(output_dir, 'corrected_lseqmdl'))
    lseqmdl_corrected_precip_path = raw_lseqmdl.replace('{output_dir}', output_dir)

    raw_models = out_dirs.get('trained_models', os.path.join(output_dir, 'trained_models'))
    trained_models_path = raw_models.replace('{output_dir}', output_dir)

    # --- Output filename template ---
    output_filename_template = cfg.get('output_filename_template', output_filename_template)

    # --- Metrics/quality output paths ---
    raw_metrics = out_dirs.get('metrics', os.path.join(output_dir, 'metrics_{method}'))
    metrics_path_template = raw_metrics.replace('{output_dir}', output_dir)

    raw_quality = out_dirs.get('quality', os.path.join(output_dir, 'quality_{method}'))
    quality_path_template = raw_quality.replace('{output_dir}', output_dir)

    # --- Statistical parameters ---
    stats = cfg.get('statistical_params', {})
    N_SPLITS_GPD_CROSSVALIDATE = stats.get('n_splits_gpd_crossvalidate', N_SPLITS_GPD_CROSSVALIDATE)
    GPD_THRESHOLD_PERCENTILE = stats.get('gpd_threshold_percentile', GPD_THRESHOLD_PERCENTILE)
    UPPER_CAP_THRESHOLD_PERCENTILE = stats.get('upper_cap_threshold_percentile', UPPER_CAP_THRESHOLD_PERCENTILE)

    # --- Metrics parameters ---
    metrics_cfg = cfg.get('metrics', {})
    WET_DAY_THRESHOLD = metrics_cfg.get('wet_day_threshold', WET_DAY_THRESHOLD)

    # --- Quality assessment parameters ---
    qa_cfg = cfg.get('quality_assessment', {})
    qa_weights = qa_cfg.get('component_weights', {})
    if qa_weights:
        QA_COMPONENT_WEIGHTS = qa_weights
    qa_thresholds = qa_cfg.get('categorical_thresholds', {})
    if qa_thresholds:
        QA_CATEGORICAL_THRESHOLDS = qa_thresholds

    # --- Deep learning parameters ---
    dl = cfg.get('deep_learning', {})
    DL_EPOCHS = dl.get('epochs', DL_EPOCHS)
    DL_BATCH_SIZE = dl.get('batch_size', DL_BATCH_SIZE)
    DL_VALIDATION_SPLIT = dl.get('validation_split', DL_VALIDATION_SPLIT)
    DL_EARLY_STOPPING_PATIENCE = dl.get('early_stopping_patience', DL_EARLY_STOPPING_PATIENCE)
    DL_DROPOUT_RATE_1 = dl.get('dropout_rate_1', DL_DROPOUT_RATE_1)
    DL_DROPOUT_RATE_2 = dl.get('dropout_rate_2', DL_DROPOUT_RATE_2)
    DL_DROPOUT_RATE_DENSE = dl.get('dropout_rate_dense', DL_DROPOUT_RATE_DENSE)
    DL_FILTER_SIZE_1 = tuple(dl.get('filter_size_1', list(DL_FILTER_SIZE_1)))
    DL_FILTER_SIZE_2 = tuple(dl.get('filter_size_2', list(DL_FILTER_SIZE_2)))
    DL_NUM_FILTERS_1 = dl.get('num_filters_1', DL_NUM_FILTERS_1)
    DL_NUM_FILTERS_2 = dl.get('num_filters_2', DL_NUM_FILTERS_2)
    DL_DENSE_LAYER_SIZE = dl.get('dense_layer_size', DL_DENSE_LAYER_SIZE)
    DL_OPTIMIZER = dl.get('optimizer', DL_OPTIMIZER)
    # random_seed: integer for reproducibility, or null to keep the original
    # stochastic behaviour (different weights + dropout on every run).
    DL_RANDOM_SEED = dl.get('random_seed', DL_RANDOM_SEED)
    DL_BLEND_ALPHA = dl.get('blend_alpha', DL_BLEND_ALPHA)

    # --- Station density confidence mask ---
    density = cfg.get('station_density', {})
    USE_CONFIDENCE_MASK = density.get('use_confidence_mask', USE_CONFIDENCE_MASK)
    raw_station = density.get('station_file')
    if raw_station:
        STATION_FILE = raw_station.replace('{main_dir}', main_dir).replace('{input_dir}', input_dir)
    raw_conf = density.get('confidence_mask_file')
    if raw_conf:
        CONFIDENCE_MASK_FILE = raw_conf.replace('{output_dir}', output_dir).replace('{main_dir}', main_dir)
    DENSITY_CPC_RESOLUTION = density.get('cpc_resolution', DENSITY_CPC_RESOLUTION)
    DENSITY_SMOOTHING_SIGMA = density.get('smoothing_sigma', DENSITY_SMOOTHING_SIGMA)
    DENSITY_SATURATION_COUNT = density.get('saturation_count', DENSITY_SATURATION_COUNT)
    lat_range = density.get('lat_range')
    if lat_range:
        DENSITY_LAT_RANGE = tuple(lat_range)
    else:
        DENSITY_LAT_RANGE = AOI_LAT_RANGE  # Default to AOI if not set
    lon_range = density.get('lon_range')
    if lon_range:
        DENSITY_LON_RANGE = tuple(lon_range)
    else:
        DENSITY_LON_RANGE = AOI_LON_RANGE  # Default to AOI if not set

    # --- Station validation ---
    stn_val = cfg.get('station_validation', {})
    raw_data = stn_val.get('station_data_file')
    if raw_data:
        STATION_DATA_FILE = raw_data.replace(
            '{main_dir}', main_dir
        ).replace('{input_dir}', input_dir)
    raw_val_out = stn_val.get('output_dir')
    if raw_val_out:
        STATION_VALIDATION_OUTPUT_DIR = raw_val_out.replace(
            '{output_dir}', output_dir
        ).replace('{main_dir}', main_dir)
    else:
        STATION_VALIDATION_OUTPUT_DIR = os.path.join(output_dir, 'station_validation')
    MISSING_SENTINEL = stn_val.get('missing_sentinel', MISSING_SENTINEL)
    MIN_VALID_DAYS = stn_val.get('min_valid_days', MIN_VALID_DAYS)

    # --- Region mapping ---
    region_cfg = cfg.get('region_mapping', {})
    p2r = region_cfg.get('province_to_region')
    if p2r:
        REGION_MAPPING = p2r
    island_order = region_cfg.get('island_order')
    if island_order:
        ISLAND_ORDER = island_order
    elif REGION_MAPPING:
        # Derive from unique region values, preserving insertion order
        seen = {}
        for r in REGION_MAPPING.values():
            seen.setdefault(r, None)
        ISLAND_ORDER = list(seen.keys())

    # --- Visualisation / Admin Boundaries ---
    vis = cfg.get('visualisation', {})
    raw_bdir = vis.get('boundary_dir')
    if raw_bdir:
        BOUNDARY_DIR = (raw_bdir
                        .replace('{main_dir}', main_dir)
                        .replace('{input_dir}', input_dir))
    BOUNDARIES = vis.get('boundaries', []) or []
    MAP_EXTENT_PAD = float(vis.get('map_extent_pad', MAP_EXTENT_PAD))

    # --- Runtime settings ---
    runtime = cfg.get('runtime', {})
    INTERACTIVE = runtime.get('interactive', INTERACTIVE)
    EXISTING_FILE_ACTION = runtime.get('existing_file_action', EXISTING_FILE_ACTION)
    EXISTING_MODEL_ACTION = runtime.get('existing_model_action', EXISTING_MODEL_ACTION)

    # --- NetCDF engine ---
    # Re-detect if not already set (guards against partial module imports)
    if NETCDF_ENGINE is None:
        NETCDF_ENGINE = _detect_netcdf_engine()

    # Create output directories
    _ensure_output_directories()

    logging.info("Configuration initialized successfully.")
    logging.info(f"  Main directory: {main_dir}")
    logging.info(f"  Input directory: {input_dir}")
    logging.info(f"  Output directory: {output_dir}")
    logging.info(f"  Interactive mode: {INTERACTIVE}")

    return cfg


def _ensure_output_directories():
    """Create output directories if they don't already exist."""
    output_directories = [
        output_dir,
        ls_corrected_precip_path,
        lseqm_corrected_precip_path,
        lseqmdl_corrected_precip_path,
        trained_models_path,
    ]
    if STATION_VALIDATION_OUTPUT_DIR:
        output_directories.append(STATION_VALIDATION_OUTPUT_DIR)
    for directory in output_directories:
        os.makedirs(directory, exist_ok=True)
