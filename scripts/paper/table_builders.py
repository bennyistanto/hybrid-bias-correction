"""
table_builders — regenerate paper Tables 4, 5, 6, 7 with corrected framing.

Headline changes vs the existing nb07 builders:

* Dekad-aggregated r/RMSE/NSE are the headline temporal-skill numbers
  (Table 4 row group "dekad resolution"). Daily numbers are kept as a
  secondary "daily resolution (timing-bound)" row group.
* KS distributional alignment is reported as **fraction of pixels with
  p > 0.05** rather than the median p-value, with the median p as a
  secondary diagnostic with the correct unit ("probability", not "%").
* Per-station Table 7 follows the same convention.

All builders return tidy pandas DataFrames; nb07 calls these and renders
markdown / LaTeX.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr

from .paper_helpers import METHODS, METHOD_LABELS
from .dekad_aggregation import (
    compute_dekad_grid_metrics, compute_dekad_station_metrics,
)


# +++++++++++++++++++++++++++++++++++++++++
# Helpers
# +++++++++++++++++++++++++++++++++++++++++

_HEADLINE_GRID = [
    ('relative_bias', 'Relative Bias', 'ratio'),
    ('pearson_correlation', 'Pearson r', '—'),
    ('rmse', 'RMSE', 'mm/dekad'),
    ('nse', 'NSE', '—'),
    ('stdev_ratio', 'σ ratio', '—'),
    ('pod', 'POD', '—'),
    ('far', 'FAR', '—'),
    ('csi', 'CSI', '—'),
    ('ks_stat', 'KS statistic', '—'),
]


def _spatial_median(ds, var):
    """Spatial median of a 2D metric variable, ignoring NaNs."""
    if var not in ds:
        return np.nan
    return float(np.nanmedian(ds[var].values))


def _ks_passing_fraction(ds, alpha=0.05, var='ks_pvalue'):
    """Fraction of land pixels with KS p > alpha."""
    if var not in ds:
        return np.nan
    arr = ds[var].values
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float((finite > alpha).mean())


# +++++++++++++++++++++++++++++++++++++++++
# Table 4 — Grid-vs-grid headline (dekad)
# +++++++++++++++++++++++++++++++++++++++++

def build_table4_grid(ref_label='cpc', threshold=1.0, apply_mask=True,
                      include_ks_passing=True):
    """
    Build Table 4: per-method spatial-median metrics on dekad-aggregated
    grid-vs-grid evaluation, plus KS pixel-passing fraction.

    Parameters
    ----------
    ref_label : {'cpc', 'imergl', 'imergf'}
    threshold : float
        Wet-dekad threshold (mm/dekad).
    apply_mask : bool
    include_ks_passing : bool
        Include 'KS pixels p>0.05 (%)' row.

    Returns
    -------
    pandas.DataFrame
        Index = metric labels, columns = method labels.
    """
    rows = {}
    ks_passing = {}
    for method, label in zip(METHODS, METHOD_LABELS):
        logging.info(f"Computing dekad grid metrics for {label}...")
        ds = compute_dekad_grid_metrics(
            method=method, ref_label=ref_label,
            apply_mask=apply_mask, threshold=threshold,
        )
        col = {}
        for var, pretty, _unit in _HEADLINE_GRID:
            col[pretty] = _spatial_median(ds, var)
        # KS p-value median (raw probability, not %)
        col['KS p-value (median, prob.)'] = _spatial_median(ds, 'ks_pvalue')
        rows[label] = col
        if include_ks_passing:
            ks_passing[label] = 100.0 * _ks_passing_fraction(ds, alpha=0.05)

    df = pd.DataFrame(rows)
    if include_ks_passing:
        df.loc['KS pixels p>0.05 (%)'] = pd.Series(ks_passing)
    return df


# +++++++++++++++++++++++++++++++++++++++++
# Table 7 — Station-vs-grid (dekad)
# +++++++++++++++++++++++++++++++++++++++++

def build_table7_station(threshold=1.0, min_valid_dekads=10):
    """
    Build Table 7: per-station median metrics on dekad-aggregated
    station-vs-grid evaluation, plus station-level KS-passing fraction.

    Returns
    -------
    pandas.DataFrame
        Index = metric labels, columns = method labels.
    pandas.DataFrame
        Raw per-station, per-method metrics (long form), for downstream
        figures (boxplots, scatter, etc.).
    """
    summaries = {}
    raw = {}
    for method, label in zip(METHODS, METHOD_LABELS):
        logging.info(f"Computing dekad station metrics for {label}...")
        df = compute_dekad_station_metrics(
            method=method, threshold=threshold,
            min_valid_dekads=min_valid_dekads,
        )
        raw[label] = df
        col = {}
        for var, pretty, _unit in _HEADLINE_GRID:
            col[pretty] = float(np.nanmedian(df[var].values)) if var in df else np.nan
        col['KS p-value (median, prob.)'] = float(np.nanmedian(df['ks_pvalue'].values))
        # Fraction of stations passing KS at α=0.05
        ks = df['ks_pvalue'].values
        finite = ks[np.isfinite(ks)]
        col['KS stations p>0.05 (%)'] = float((finite > 0.05).mean() * 100.0) if finite.size else np.nan
        summaries[label] = col

    summary_df = pd.DataFrame(summaries)
    raw_long = pd.concat(
        {label: df for label, df in raw.items()},
        names=['method', 'station_id'],
    )
    return summary_df, raw_long


# +++++++++++++++++++++++++++++++++++++++++
# KS pixel-passing fraction (alone, for §4.1 prose)
# +++++++++++++++++++++++++++++++++++++++++

def build_ks_pixel_passing_fraction(ref_label='cpc',
                                    alphas=(0.05, 0.10),
                                    apply_mask=True):
    """
    For each method, compute the fraction of land pixels whose KS p-value
    exceeds each alpha threshold, on dekad-aggregated grids.

    This is the headline distributional-alignment number for §4.1.

    Returns
    -------
    pandas.DataFrame
        Index = method labels, columns = one per alpha (e.g., 'p>0.05 (%)').
    """
    rows = {}
    for method, label in zip(METHODS, METHOD_LABELS):
        ds = compute_dekad_grid_metrics(
            method=method, ref_label=ref_label,
            apply_mask=apply_mask, threshold=1.0,
        )
        rows[label] = {
            f'p>{a:.2f} (%)': 100.0 * _ks_passing_fraction(ds, alpha=a)
            for a in alphas
        }
    return pd.DataFrame(rows).T
