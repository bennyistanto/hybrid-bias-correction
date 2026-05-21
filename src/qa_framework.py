"""
Module: qa_framework.py

Quality Assessment (QA) framework for bias-corrected precipitation products.

This module computes composite quality indices from the raw verification metrics
produced by ``metrics.py``. It implements a three-tier quality assessment:

1. **Basic Statistical Quality** (config weight 0.4):
   Combines Relative Bias, RMSE, and NSE into a single normalized score [0, 1].
   Note: Pearson Correlation is intentionally excluded here because it is
   already embedded within NSE (Gupta et al., 2009). Including both would
   double-count the linear association component.

2. **Distribution Quality** (config weight 0.3):
   Evaluates how well the corrected product preserves the full precipitation
   distribution: extreme percentiles (p90/p95/p99) with highest weight
   (primary target of GPD tail adjustment), general percentiles (p25/p50/p75),
   variability (std dev ratio), and KS p-value (formal distribution test).

3. **Temporal Quality** (config weight 0.3):
   Assesses preservation of temporal patterns: CSI for balanced event detection
   skill, POD/FAR for detection vs false alarm trade-off, and dry spell length
   for sequencing. CSI is preferred over correlation to avoid double-counting
   with NSE in Basic Statistical Quality.

These three components combine into:
- **Continuous Quality Index (CQI)**: weighted average, range [0, 1].
- **Categorical Quality**: Poor (<0.4), Fair (0.4-0.6), Good (0.6-0.8),
  Excellent (>=0.8).
- **Confidence Level**: reliability estimate based on metric consistency,
  sample size, and distribution agreement (KS p-value, where HIGH p-value
  indicates similar distributions and thus high confidence).

References
----------
- WMO (2017), Guidelines on the Calculation of Climate Normals.
- Gupta, H. V. et al. (2009), Decomposition of the mean squared error and
  NSE performance criteria. J. Hydrology, 377(1-2), 80-91.
- Wilks, D.S. (2011), Statistical Methods in the Atmospheric Sciences, 3rd ed.
- Entekhabi et al. (2010), Performance Metrics for Soil Moisture Retrievals.
- Ebert, E. (2007), Methods for verifying satellite precipitation estimates.

Author
------
Benny Istanto
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""
import os
import numpy as np
import xarray as xr
import pandas as pd
import logging

from .config import NETCDF_ENGINE
from .utility import set_user_decision

# +++++++++++++++++++++++++++++++++++++++++
# CF-1.8 Encoding for QA Output
# +++++++++++++++++++++++++++++++++++++++++

CF18_QUALITY = {
    'basic_statistical_quality': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan},
    'distribution_quality': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan},
    'temporal_quality': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan},
    'continuous_quality': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan},
    'categorical_quality': {'dtype': 'int32', 'zlib': True, '_FillValue': -9999},
    'confidence_level': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan},
}


# +++++++++++++++++++++++++++++++++++++++++
# Component Quality Scores
# +++++++++++++++++++++++++++++++++++++++++

def calculate_basic_statistical_quality(metrics, weights=None):
    """
    Calculate basic statistical quality score combining fundamental metrics.

    Each metric is normalized to [0, 1] where 1 = perfect:
    - Relative Bias: 1 - min(|RB|, 1). Penalizes both over- and under-estimation.
    - RMSE: exp(-RMSE / 5). Exponential decay; RMSE of 5 mm/day scores ~0.37.
    - NSE: clipped to [0, 1]. Values < 0 (worse than climatology) score 0.

    Note: Pearson Correlation is intentionally excluded here because it is
    already embedded within NSE (Nash-Sutcliffe Efficiency). Including both
    would double-count the linear association component. NSE is preferred
    because it integrates correlation, bias, and variability into a single
    measure (Gupta et al., 2009).

    Parameters
    ----------
    metrics : xarray.Dataset
        Dataset containing at least: relative_bias, rmse, nse.
    weights : dict, optional
        Custom weights for each metric. Keys: 'relative_bias',
        'rmse', 'nse'. Must sum to 1.0.
        Default: RB=0.30, RMSE=0.30, NSE=0.40.

    Returns
    -------
    xarray.DataArray
        Basic statistical quality score, range [0, 1].
    """
    if weights is None:
        weights = {
            'relative_bias': 0.30,
            'rmse': 0.30,
            'nse': 0.40,
        }

    # Normalize each metric to [0, 1]
    rb_score = 1 - np.minimum(np.abs(metrics['relative_bias']), 1)
    rmse_score = np.exp(-metrics['rmse'] / 5)
    nse_score = np.maximum(np.minimum(metrics['nse'], 1), 0)

    basic_score = (
        weights['relative_bias'] * rb_score +
        weights['rmse'] * rmse_score +
        weights['nse'] * nse_score
    ).astype('float32')

    basic_score.attrs.update({
        'long_name': 'Basic Statistical Quality Score',
        'units': 'unitless',
        'valid_range': [0, 1],
        'description': 'Weighted combination of RB, RMSE, NSE scores',
    })

    return basic_score


def calculate_distribution_quality(metrics, weights=None):
    """
    Calculate distribution quality score from percentile matching, variability,
    and formal distribution testing.

    Evaluates four aspects:
    1. **Extreme percentile matching** (p90, p95, p99): how well the upper
       tail is reproduced. The 99th percentile has higher weight because
       extremes are the primary target of the GPD tail adjustment. This is
       the most important sub-component for our EQM+GPD workflow.
    2. **General percentile matching** (p25, p50, p75): how well the bulk
       distribution is reproduced. Score = 1 - |test - ref| / (ref + 0.1).
    3. **Variability preservation**: |1 - stdev_ratio|. A ratio of 1 means
       perfect variability preservation.
    4. **KS p-value**: formal two-sample Kolmogorov-Smirnov test. A high
       p-value means no statistical evidence that the distributions differ,
       indicating successful correction. This provides a rigorous statistical
       complement to the empirical percentile comparisons.

    Parameters
    ----------
    metrics : xarray.Dataset
        Dataset containing percentile variables (p25_ref, p25_test, etc.),
        stdev_ratio, and ks_pvalue.
    weights : dict, optional
        Weights for 'extreme_percentiles', 'general_percentiles',
        'variability', 'ks_test'. Must sum to 1.0.
        Default: 0.45, 0.25, 0.15, 0.15.

    Returns
    -------
    xarray.DataArray
        Distribution quality score, range [0, 1].
    """
    if weights is None:
        weights = {
            'extreme_percentiles': 0.45,
            'general_percentiles': 0.25,
            'variability': 0.15,
            'ks_test': 0.15,
        }

    # Extreme percentile matching (p90, p95, p99) - most important for GPD/DL
    extreme_pairs = [
        ('p90_ref', 'p90_test', 0.3),  # 90th: 30% weight
        ('p95_ref', 'p95_test', 0.3),  # 95th: 30% weight
        ('p99_ref', 'p99_test', 0.4),  # 99th: 40% weight - critical for extremes
    ]
    extreme_score = sum(
        w * (1 - np.minimum(np.abs(metrics[t] - metrics[r]) / (metrics[r] + 0.1), 1))
        for r, t, w in extreme_pairs
    )

    # General percentile matching (p25, p50, p75)
    general_pairs = [
        ('p25_ref', 'p25_test', 0.3),  # 25th percentile: 30% weight
        ('p50_ref', 'p50_test', 0.4),  # 50th (median): 40% weight - most important
        ('p75_ref', 'p75_test', 0.3),  # 75th percentile: 30% weight
    ]
    general_score = sum(
        w * (1 - np.minimum(np.abs(metrics[t] - metrics[r]) / (metrics[r] + 0.1), 1))
        for r, t, w in general_pairs
    )

    # Variability preservation: ideal stdev_ratio = 1
    var_score = 1 - np.minimum(np.abs(1 - metrics['stdev_ratio']), 1)

    # KS p-value: high p-value = no evidence distributions differ = good
    # Clipped to [0, 1] for safety (already in that range by definition)
    ks_score = np.maximum(np.minimum(metrics['ks_pvalue'], 1), 0)

    dist_score = (
        weights['extreme_percentiles'] * extreme_score +
        weights['general_percentiles'] * general_score +
        weights['variability'] * var_score +
        weights['ks_test'] * ks_score
    )

    dist_score.attrs.update({
        'long_name': 'Distribution Quality Score',
        'units': 'unitless',
        'valid_range': [0, 1],
        'description': 'Percentile matching + variability + KS test score',
    })

    return dist_score


def calculate_temporal_quality(metrics, weights=None):
    """
    Calculate temporal quality score for pattern and event preservation.

    Evaluates three aspects:
    1. **CSI (Critical Success Index)**: the most balanced categorical metric,
       simultaneously penalizing both misses and false alarms. Unlike POD
       (ignores false alarms) or FAR (ignores misses), CSI provides a single
       measure of overall event detection skill (Wilks, 2011).
    2. **Event timing**: weighted combination of POD and (1-FAR).
       POD measures detection of real events; (1-FAR) penalizes false alarms.
    3. **Dry spell preservation**: how well dry periods are maintained.
       Score = 1 - |dsl_test - dsl_ref| / (dsl_ref + epsilon).

    Note: Pearson Correlation is intentionally excluded here to avoid
    double-counting with Basic Statistical Quality, where it is embedded
    within NSE. CSI is used instead as it provides independent information
    about categorical event detection accuracy.

    Parameters
    ----------
    metrics : xarray.Dataset
        Dataset containing: csi, pod, far, dsl_ref, dsl_test.
    weights : dict, optional
        Weights for 'csi', 'event_timing', 'spell_preservation'.
        Must sum to 1.0. Default: 0.4, 0.3, 0.3.

    Returns
    -------
    xarray.DataArray
        Temporal quality score, range [0, 1].
    """
    if weights is None:
        weights = {
            'csi': 0.4,
            'event_timing': 0.3,
            'spell_preservation': 0.3,
        }

    # CSI: already in [0, 1] where 1 = perfect
    csi_score = np.maximum(np.minimum(metrics['csi'].values, 1), 0)

    # Event timing: 0.6 * POD + 0.4 * (1 - FAR)
    event_score = 0.6 * metrics['pod'].values + 0.4 * (1 - metrics['far'].values)

    # Dry spell preservation
    dsl_ref = metrics['dsl_ref'].fillna(0).values.astype('float64')
    dsl_test = metrics['dsl_test'].fillna(0).values.astype('float64')

    # Handle timedelta-encoded values from some xarray operations
    if dsl_ref.dtype.kind == 'm':  # timedelta
        ns_per_day = np.float64(8.64e13)
        dsl_ref = dsl_ref.astype('float64') / ns_per_day
        dsl_test = dsl_test.astype('float64') / ns_per_day

    spell_score = 1 - np.minimum(
        np.abs(dsl_test - dsl_ref) / (dsl_ref + 1e-6), 1
    )

    # Weighted combination
    temporal_score = xr.DataArray(
        weights['csi'] * csi_score +
        weights['event_timing'] * event_score +
        weights['spell_preservation'] * spell_score,
        coords=metrics['csi'].coords,
        dims=metrics['csi'].dims,
    )

    temporal_score.attrs.update({
        'long_name': 'Temporal Quality Score',
        'units': 'unitless',
        'valid_range': [0, 1],
        'description': 'CSI + event detection + dry spell preservation',
    })

    return temporal_score


# +++++++++++++++++++++++++++++++++++++++++
# Composite Quality Indices
# +++++++++++++++++++++++++++++++++++++++++

def calculate_overall_quality(basic_score, dist_score, temporal_score,
                              component_weights=None, categorical_thresholds=None):
    """
    Calculate Continuous Quality Index (CQI) and categorical classification.

    Parameters
    ----------
    basic_score : xarray.DataArray
        Basic statistical quality score [0, 1].
    dist_score : xarray.DataArray
        Distribution quality score [0, 1].
    temporal_score : xarray.DataArray
        Temporal quality score [0, 1].
    component_weights : dict, optional
        Weights for 'basic_stats', 'distribution', 'temporal'.
        Default: 0.35, 0.35, 0.30.
    categorical_thresholds : dict, optional
        Thresholds for 'excellent', 'good', 'fair'.
        Default: 0.8, 0.6, 0.4.

    Returns
    -------
    continuous_quality : xarray.DataArray
        CQI score, range [0, 1].
    categorical_quality : xarray.DataArray
        Classification: 4=Excellent, 3=Good, 2=Fair, 1=Poor, -9999=NoData.
    """
    if component_weights is None:
        component_weights = {'basic_stats': 0.35, 'distribution': 0.35, 'temporal': 0.30}
    if categorical_thresholds is None:
        categorical_thresholds = {'excellent': 0.8, 'good': 0.6, 'fair': 0.4}

    # Continuous Quality Index
    continuous_quality = (
        component_weights['basic_stats'] * basic_score +
        component_weights['distribution'] * dist_score +
        component_weights['temporal'] * temporal_score
    ).astype('float32')

    # Categorical classification
    cat = xr.full_like(continuous_quality, 1, dtype='int32')
    cat = xr.where(continuous_quality >= categorical_thresholds['fair'], 2, cat)
    cat = xr.where(continuous_quality >= categorical_thresholds['good'], 3, cat)
    cat = xr.where(continuous_quality >= categorical_thresholds['excellent'], 4, cat)
    cat = xr.where(np.isnan(continuous_quality), -9999, cat).astype('int32')

    continuous_quality.attrs.update({
        'long_name': 'Continuous Quality Index',
        'units': 'unitless',
        'valid_range': [0, 1],
        'description': 'Weighted average of basic, distribution, and temporal scores',
    })
    cat.attrs.update({
        'long_name': 'Categorical Quality Classification',
        'units': 'category',
        'flag_values': [1, 2, 3, 4],
        'flag_meanings': 'poor fair good excellent',
    })

    return continuous_quality, cat


def calculate_confidence(metrics, continuous_quality):
    """
    Estimate confidence in the quality assessment.

    Confidence is derived from:
    - **Metric consistency**: agreement among different metrics (low std = high
      confidence). If bias, NSE, POD, and FAR all agree, the assessment
      is more trustworthy.
    - **Distribution agreement**: KS-test p-value used directly (NOT inverted).
      A HIGH p-value means the reference and test distributions are NOT
      statistically different, indicating the correction successfully matched
      the reference distribution, thus warranting HIGH confidence.
    - **Sample size** (timeseries only): fraction of non-NaN years.

    For single dekad: Confidence = 0.6 * consistency + 0.4 * dist_agreement.
    For timeseries:   Confidence = 0.4 * sample + 0.3 * dist_agreement + 0.3 * consistency.

    Parameters
    ----------
    metrics : xarray.Dataset
        Raw metrics dataset.
    continuous_quality : xarray.DataArray
        CQI score (used for shape reference).

    Returns
    -------
    xarray.DataArray
        Confidence level, range [0, 1].
    """
    try:
        # Metric consistency: stdev across normalized metrics
        # Using RB, NSE, POD, 1-FAR - all normalized to [0, 1] where 1 = good
        arr_rb = np.maximum(1 - np.abs(metrics['relative_bias']), 0)
        arr_nse = np.maximum(np.minimum(metrics['nse'], 1), 0)
        arr_pod = metrics['pod']
        arr_far = 1 - metrics['far']

        stacked = xr.concat([arr_rb, arr_nse, arr_pod, arr_far], dim='_metric')
        consistency = 1 - stacked.std(dim='_metric', skipna=True)

        # Distribution agreement: KS p-value used DIRECTLY
        # High p-value = distributions are similar = high confidence
        # (Previously inverted as 1-p_value, which was incorrect: it gave
        #  high confidence when distributions were MOST different)
        dist_agreement = np.maximum(np.minimum(metrics['ks_pvalue'], 1), 0)

        if 'time' not in metrics.dims:
            # Single dekad
            confidence = 0.6 * consistency + 0.4 * dist_agreement
        else:
            # Timeseries: include sample size factor
            valid_points = ~np.isnan(metrics['relative_bias'])
            sample_factor = valid_points.sum('time') / len(metrics.time)
            confidence = (
                0.4 * sample_factor +
                0.3 * dist_agreement +
                0.3 * consistency
            )

        confidence.attrs.update({
            'long_name': 'Quality Assessment Confidence',
            'units': 'unitless',
            'valid_range': [0, 1],
            'description': 'Reliability of the quality assessment',
        })
        return confidence

    except Exception as e:
        logging.error(f"Error computing confidence: {e}")
        default = xr.full_like(continuous_quality, 0.5)
        default.attrs.update({
            'long_name': 'Quality Assessment Confidence',
            'units': 'unitless', 'valid_range': [0, 1],
            'description': 'Default confidence (computation error)',
        })
        return default


# +++++++++++++++++++++++++++++++++++++++++
# Full Pipeline and I/O
# +++++++++++++++++++++++++++++++++++++++++

def compute_quality_assessment(metrics_ds, component_weights=None,
                               categorical_thresholds=None):
    """
    Run the full quality assessment pipeline on a metrics dataset.

    This is a convenience function that chains all component calculations
    and returns a single Dataset with all quality variables.

    Parameters
    ----------
    metrics_ds : xarray.Dataset
        Raw metrics from ``metrics.compute_dekad_metrics_timeseries``
        or ``metrics.compute_single_dekad_metrics``.
    component_weights : dict, optional
        Weights for basic/distribution/temporal components.
    categorical_thresholds : dict, optional
        Thresholds for categorical classification.

    Returns
    -------
    xarray.Dataset
        Quality assessment dataset with variables:
        basic_statistical_quality, distribution_quality, temporal_quality,
        continuous_quality, categorical_quality, confidence_level.
    """
    basic = calculate_basic_statistical_quality(metrics_ds)
    dist = calculate_distribution_quality(metrics_ds)
    temporal = calculate_temporal_quality(metrics_ds)

    continuous, categorical = calculate_overall_quality(
        basic, dist, temporal, component_weights, categorical_thresholds
    )

    confidence = calculate_confidence(metrics_ds, continuous)

    quality_ds = xr.Dataset({
        'basic_statistical_quality': basic,
        'distribution_quality': dist,
        'temporal_quality': temporal,
        'continuous_quality': continuous,
        'categorical_quality': categorical,
        'confidence_level': confidence,
    })

    return quality_ds


def save_quality_assessment(quality_ds, out_file,
                            description="Bias Correction Quality Assessment"):
    """
    Save quality assessment results to CF-1.8 compliant NetCDF.

    Parameters
    ----------
    quality_ds : xarray.Dataset
        Dataset from ``compute_quality_assessment``.
    out_file : str
        Output file path.
    description : str, optional
        Title for the dataset global attribute.

    Returns
    -------
    str or None
        Path to saved file, or None if skipped.
    """
    if os.path.exists(out_file):
        logging.info(f"File {out_file} already exists.")
        decision = set_user_decision()
        if decision == 'S':
            logging.info(f"Skipping {out_file}")
            return None
        elif decision == 'A':
            from .io import BiasCorrectAbort
            raise BiasCorrectAbort("User chose to abort.")

    quality_ds.attrs.update({
        'title': description,
        'Conventions': 'CF-1.8',
        'institution': 'Bogor Agricultural University',
        'source': 'Bias Correction Quality Assessment',
        'references': 'WMO Guidelines for Precipitation Verification',
        'history': f'Created on {pd.Timestamp.now()}',
        'creator_name': 'Benny Istanto',
        'creator_role': 'Climate Geographer',
        'creator_email': 'bennyistanto@apps.ipb.ac.id',
    })

    if 'lat' in quality_ds.coords:
        quality_ds['lat'].attrs.update({
            'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y',
        })
    if 'lon' in quality_ds.coords:
        quality_ds['lon'].attrs.update({
            'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X',
        })

    encoding = {var: CF18_QUALITY[var] for var in quality_ds.data_vars if var in CF18_QUALITY}

    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        quality_ds.to_netcdf(out_file, engine=NETCDF_ENGINE, encoding=encoding)
        logging.info(f"Saved quality assessment to {out_file}")
        return out_file
    except IOError as e:
        logging.error(f"Failed to save {out_file}: {e}")
        return None


# +++++++++++++++++++++++++++++++++++++++++
# Main Orchestration Pipeline
# +++++++++++++++++++++++++++++++++++++++++

def run_qa_pipeline(month, dekad, mode='single'):
    """
    Run quality assessment for all 9 reference-vs-test combinations.

    Loads each metrics NetCDF produced by ``metrics.run_metrics_pipeline``,
    runs :func:`compute_quality_assessment`, and saves the result via
    :func:`save_quality_assessment`.

    Combinations: {CPC, IMERGL, IMERGF} × {LS, LSEQM, LSEQM+DL}.

    Parameters
    ----------
    month : int
        Month number (1-12).
    dekad : int
        Dekad number (1, 2, or 3).
    mode : str, optional
        ``'single'`` (default) - aggregated across all years, dims=(lat, lon).
        ``'timeseries'`` - per-year quality, dims=(time, lat, lon).

    Returns
    -------
    list of str
        Paths to the saved quality NetCDF files (None for skipped combos).
    """
    from . import config

    if mode not in ('timeseries', 'single'):
        raise ValueError(f"mode must be 'timeseries' or 'single', got '{mode}'")

    # Format strings matching run_metrics_pipeline naming convention
    month_str = f"{month:02d}"
    if dekad == 1:
        dekad_str = "01"
    elif dekad == 2:
        dekad_str = "11"
    else:
        dekad_str = "21"

    metrics_prefix = 'metricsts' if mode == 'timeseries' else 'metricssd'
    quality_prefix = 'qualityts' if mode == 'timeseries' else 'qualitysd'

    ref_labels = ['cpc', 'imergl', 'imergf']
    method_labels = ['ls', 'lseqm', 'lseqmdl']

    output_files = []
    combo_num = 0

    for ref_label in ref_labels:
        for method in method_labels:
            combo_num += 1
            test_label = f"imergl_{method}"

            # Resolve paths from config templates
            metrics_dir = config.metrics_path_template.replace('{method}', method)
            quality_dir = config.quality_path_template.replace('{method}', method)

            # Metrics input file (produced by run_metrics_pipeline)
            metrics_fname = (
                f"{config.FILENAME_PREFIX}_{metrics_prefix}_{ref_label}_{test_label}"
                f"_month{month_str}_dekad{dekad_str}.nc4"
            )
            metrics_fpath = os.path.join(metrics_dir, metrics_fname)

            if not os.path.isfile(metrics_fpath):
                logging.info(
                    f"[{combo_num}/9] Metrics not found: {metrics_fname} - skipping"
                )
                output_files.append(None)
                continue

            logging.info(f"[{combo_num}/9] {ref_label} vs {test_label} ({mode})")

            try:
                # with-block closes the file on both success and exception
                # paths. The previous version closed only on success; if
                # compute_quality_assessment or save_quality_assessment
                # raised, the NetCDF handle leaked.
                with xr.open_dataset(
                    metrics_fpath, engine=NETCDF_ENGINE,
                    decode_timedelta=False,
                ) as metrics_ds:
                    # Run full quality assessment (use config weights if set)
                    quality_ds = compute_quality_assessment(
                        metrics_ds,
                        component_weights=config.QA_COMPONENT_WEIGHTS,
                    )

                    # Build output path
                    quality_fname = (
                        f"{config.FILENAME_PREFIX}_{quality_prefix}_{ref_label}_{test_label}"
                        f"_month{month_str}_dekad{dekad_str}.nc4"
                    )
                    os.makedirs(quality_dir, exist_ok=True)
                    quality_fpath = os.path.join(quality_dir, quality_fname)

                    desc = (
                        f"{ref_label.upper()} vs {test_label.upper()} "
                        f" - {mode} quality assessment"
                    )
                    result = save_quality_assessment(
                        quality_ds, quality_fpath, description=desc
                    )
                    output_files.append(result)

                logging.info(f"  Done: {quality_fname}")

            except Exception as e:
                logging.error(
                    f"Error processing {ref_label} vs {test_label}: {e}"
                )
                output_files.append(None)

    n_saved = sum(1 for f in output_files if f is not None)
    logging.info(
        f"QA pipeline complete: {n_saved}/9 files saved ({mode} mode)"
    )

    return output_files
