"""
Hybrid Bias Correction (LSEQM+DL)

A framework for correcting biases in satellite-derived precipitation estimates
using a hybrid approach combining Linear Scaling, Empirical Quantile Mapping
with GPD tail adjustment, and Deep Learning.

Modules
-------
config
    Centralized configuration loaded from config.yml.
bias_correction
    Physical-statistical correction: LS -> EQM+GPD (Step 1).
deep_learning
    CNN training and inference for extreme value refinement (Step 2).
    Requires TensorFlow; imported lazily to allow non-DL workflows without it.
distribution_fitting
    Gamma/GPD fitting, L-moments, quantile mapping.
io
    NetCDF I/O with CF-1.8 metadata.
utility
    Land-sea masking, time alignment, user prompts.
metrics
    31 pixel-level verification metrics (continuous, categorical, percentiles).
qa_framework
    Composite quality indices (CQI, categorical, confidence).
station_density
    CPC-UNI gauge station density confidence mask.
station_validation
    Independent validation against BMKG weather station observations.

Usage
-----
    from src.config import initialize_config
    initialize_config('config.yml')

    # Step 1: Physical-statistical correction
    from src.bias_correction import lseqm
    lseqm_result, cpc_dekad = lseqm(imerg_ds, cpc_ds, month, dekad_start, dekad_end, ...)

    # Step 2: DL refinement (optional, requires TensorFlow)
    from src.deep_learning import train_bias_correction_model, apply_deeplearning_model
    model = train_bias_correction_model(lseqm_result, cpc_dekad, model_name)
    final = apply_deeplearning_model(model, lseqm_result)

Note
----
    Deep learning functions (deep_learning.py) are imported lazily because they
    depend on TensorFlow, which may not be installed in all environments.
    This allows lightweight operations (config, metrics, QA, and now also
    bias_correction) to work without TensorFlow.
"""

# ---------------------------------------------------------------------------
# Eagerly imported modules: these have no heavy dependencies beyond
# numpy, scipy, xarray, and pandas — all of which are required anyway.
# ---------------------------------------------------------------------------
from .config import initialize_config, setup_logging
from .bias_correction import lseqm, run_correction_pipeline
from .io import save_corrected_precip, get_max_day_in_month, aggregate_data_across_years, aggregate_cpc_native_for_dekad
from .utility import (
    apply_land_sea_mask,
    load_mask,
    ensure_strict_monotonic_time,
    reindex_and_align_with_monotonicity,
)
from .distribution_fitting import (
    gamma_quantile_mapping,
    gamma_quantile_mapping_precomputed,
    fit_gamma_distribution,
    calculate_l_moments,
    cross_validate_gpd,
    fit_cpc_parameters_on_native_grid,
    interpolate_cpc_params_to_imerg_grid,
)
from .station_density import (
    get_or_create_confidence_mask,
    load_confidence_mask,
    load_station_locations,
)
from .station_validation import (
    load_station_observations,
    extract_gridded_at_stations,
    compute_station_metrics,
    compute_multi_threshold_metrics,
    summarize_station_metrics,
    summarize_multi_threshold,
    merge_station_metadata,
    summarize_by_group,
    plot_station_scatter,
    save_station_validation,
    run_station_validation,
)
from .metrics import (
    compute_dekad_metrics_timeseries,
    compute_single_dekad_metrics,
    compute_pixel_metrics,
    save_metrics,
    slice_month_dekad,
    unify_cpc_for_metrics,
    build_metric_combos,
    run_metrics_pipeline,
)
from .qa_framework import (
    compute_quality_assessment,
    calculate_basic_statistical_quality,
    calculate_distribution_quality,
    calculate_temporal_quality,
    calculate_overall_quality,
    calculate_confidence,
    save_quality_assessment,
)

# ---------------------------------------------------------------------------
# Lazily imported modules: these depend on TensorFlow via deep_learning.py.
# They are only loaded when first accessed, so environments without
# TensorFlow can still use the rest of the package.
# ---------------------------------------------------------------------------


def __getattr__(name):
    """Lazy import for TensorFlow-dependent symbols.

    This function is called when an attribute is not found in the module's
    namespace. It provides lazy loading for symbols that depend on TensorFlow,
    deferring the import until the symbol is actually used.

    Lazily imported symbols:
        - train_bias_correction_model (from deep_learning)
        - apply_deeplearning_model (from deep_learning)
    """
    # Symbols from deep_learning (requires TensorFlow)
    if name == "train_bias_correction_model":
        from .deep_learning import train_bias_correction_model
        return train_bias_correction_model

    if name == "apply_deeplearning_model":
        from .deep_learning import apply_deeplearning_model
        return apply_deeplearning_model

    raise AttributeError(f"module 'src' has no attribute {name!r}")
