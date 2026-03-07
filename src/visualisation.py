"""
Module: visualisation.py

This module provides centralized visualization functions for the hybrid bias
correction project. It generates diagnostic plots for two main domains:

**QA Framework Visualization** (from notebook 04 output):
Eight plot types comparing correction quality across methods (LS, LSEQM,
LSEQM+DL) and spatial domains for Indonesia.

**Station Validation Visualization** (from notebook 05 output):
Spatial scatter maps, WMO multi-threshold performance curves, and regional
box plots summarizing independent BMKG station validation results.

The eight plot types are:
  1. CQI spatial maps -- Continuous Quality Index for each method side by side.
  2. Categorical quality maps -- Poor / Fair / Good / Excellent classification.
  3. Method improvement map -- Quality gain from LS to LSEQM+DL.
  4. Component quality maps -- Basic statistical, distribution, and temporal scores.
  5. Confidence map -- Spatial reliability of the quality assessment.
  6. CQI distribution analysis -- Empirical CDF and histograms comparing methods.
  7. Category summary -- Grouped bar chart and percentage table.
  8. Component box plots -- Box plots comparing components across methods.

Each function can run interactively (plt.show) or in batch mode (plt.close).
Figures are saved into per-plot-type sub-folders under the output directory,
organized as {quality_prefix}_{plot_type}/ (e.g., qualitysd_cqi_spatial/).

The batch orchestrator, run_qa_batch_viz, loops over all 36 dekadal periods,
calls all eight plot functions, and returns aggregated summary statistics.
print_batch_summary displays the collected results.

**Author**:
  Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""

import os
import logging

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Display labels for correction methods
TITLES_MAP = {"LS": "LS", "LSEQM": "LSEQM", "LSEQMDL": "LSEQM+DL"}

# Internal method abbreviations used in filenames
_METHOD_ABBR = {"LS": "ls", "LSEQM": "lseqm", "LSEQMDL": "lseqmdl"}

# Per-method colours (consistent across all plot types)
METHOD_COLORS = {"LS": "#1f77b4", "LSEQM": "#ff7f0e", "LSEQMDL": "#2ca02c"}

# Categorical quality colour scheme
_CAT_COLORS = ["#d32f2f", "#ff9800", "#9ccc65", "#388e3c"]
_CAT_CMAP = ListedColormap(_CAT_COLORS)
_CAT_BOUNDARIES = [0.5, 1.5, 2.5, 3.5, 4.5]
_CAT_NORM = BoundaryNorm(_CAT_BOUNDARIES, _CAT_CMAP.N)
_CAT_NAMES = ["Poor", "Fair", "Good", "Excellent"]
_CAT_VALUES = [1, 2, 3, 4]
def _cat_thresholds():
    """Return sorted QA categorical thresholds from config."""
    from src import config
    return sorted(config.QA_CATEGORICAL_THRESHOLDS.values())

# Quality component variable names + display labels
COMPONENT_VARS = [
    ("basic_statistical_quality", "Basic Statistical"),
    ("distribution_quality", "Distribution"),
    ("temporal_quality", "Temporal"),
    ("continuous_quality", "CQI"),
]

# Spatial extent — loaded lazily from config.AOI_LON_RANGE / AOI_LAT_RANGE.
def _xlim():
    """Return (lon_min, lon_max) from config."""
    from src import config
    return tuple(config.AOI_LON_RANGE)

def _ylim():
    """Return (lat_min, lat_max) from config."""
    from src import config
    return tuple(config.AOI_LAT_RANGE)

# Dekad start-day mapping
_DEKAD_START = {1: "01", 2: "11", 3: "21"}

# All 36 month/dekad combinations
ALL_DEKADS = [(m, d) for m in range(1, 13) for d in [1, 2, 3]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_period(month, dekad):
    """Return ``(month_str, dekad_str)`` for filename construction."""
    return f"{month:02d}", _DEKAD_START[dekad]


def _save_fig(fig, output_dir, plot_type, quality_prefix, month_str,
              dekad_str, dpi=150):
    """Save *fig* into ``{output_dir}/{quality_prefix}_{plot_type}/``."""
    sub_dir = os.path.join(output_dir, f"{quality_prefix}_{plot_type}")
    os.makedirs(sub_dir, exist_ok=True)
    from src import config as _cfg
    fname = (
        f"{_cfg.FILENAME_PREFIX}_qa_viz_{quality_prefix}_{plot_type}"
        f"_month{month_str}_dekad{dekad_str}.png"
    )
    fpath = os.path.join(sub_dir, fname)
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info("Saved: %s", fpath)
    return fpath


def _squeeze_time(da):
    """Select the first time step if *da* has a time dimension."""
    if "time" in da.dims:
        return da.isel(time=0)
    return da


def _finish_fig(fig, output_dir, plot_type, quality_prefix, month_str,
                dekad_str, interactive):
    """Save (if *output_dir*) and show/close *fig*."""
    path = None
    if output_dir is not None:
        path = _save_fig(fig, output_dir, plot_type, quality_prefix,
                         month_str, dekad_str)
    if interactive:
        plt.show()
    else:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_quality_data(month, dekad, quality_prefix="qualitysd",
                      ref_label="cpc", config=None):
    """Load QA NetCDF files for all three correction methods.

    Parameters
    ----------
    month : int
        Month number (1--12).
    dekad : int
        Dekad number (1, 2, or 3).
    quality_prefix : str
        ``'qualitysd'`` (single-dekad aggregated) or ``'qualityts'``
        (per-year timeseries).
    ref_label : str
        Reference dataset label (default ``'cpc'``).
    config : module, optional
        Configuration module.  If *None*, imports ``src.config``.

    Returns
    -------
    dict of {str: xarray.Dataset}
        Keyed by display name (``'LS'``, ``'LSEQM'``, ``'LSEQMDL'``).
    """
    if config is None:
        from src import config  # noqa: F811

    month_str, dekad_str = _format_period(month, dekad)
    quality_data = {}

    for display_name, method_abbr in _METHOD_ABBR.items():
        quality_dir = config.quality_path_template.replace(
            "{method}", method_abbr
        )
        test_label = f"imergl_{method_abbr}"
        fname = (
            f"{config.FILENAME_PREFIX}_{quality_prefix}_{ref_label}_{test_label}"
            f"_month{month_str}_dekad{dekad_str}.nc4"
        )
        fpath = os.path.join(quality_dir, fname)

        if os.path.exists(fpath):
            ds = xr.open_dataset(
                fpath, engine=config.NETCDF_ENGINE, decode_timedelta=False
            )
            quality_data[display_name] = ds
        else:
            logger.debug("QA file not found, skipping: %s", fpath)

    return quality_data


# ---------------------------------------------------------------------------
# Plot functions  (each returns fig; optionally saves + shows/closes)
# ---------------------------------------------------------------------------

def plot_cqi_spatial(quality_data, month, dekad,
                     quality_prefix="qualitysd", output_dir=None,
                     interactive=True):
    """CQI spatial maps — one panel per correction method.

    Parameters
    ----------
    quality_data : dict of {str: xr.Dataset}
    month, dekad : int
    quality_prefix : str
    output_dir : str or None
    interactive : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    month_str, dekad_str = _format_period(month, dekad)
    n = len(quality_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(
        1, n, figsize=(5.5 * n, 3.5),
        squeeze=False, constrained_layout=True,
    )
    axes = axes.flatten()
    im = None

    for idx, (name, ds) in enumerate(quality_data.items()):
        ax = axes[idx]
        cqi = _squeeze_time(ds["continuous_quality"])
        im = ax.pcolormesh(
            cqi.lon, cqi.lat, cqi.values,
            cmap="viridis", vmin=0, vmax=1, shading="auto",
        )
        ax.set_xlim(*_xlim())
        ax.set_ylim(*_ylim())
        ax.set_title(TITLES_MAP.get(name, name), fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if idx == 0 else "")
        ax.set_aspect("equal")

    if im is not None:
        fig.colorbar(
            im, ax=axes.tolist(), orientation="vertical",
            shrink=0.85, aspect=18, pad=0.02,
            label="CQI (0 = poor, 1 = excellent)",
        )

    fig.suptitle(
        f"Continuous Quality Index (CQI) -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )

    _finish_fig(fig, output_dir, "cqi_spatial", quality_prefix,
                month_str, dekad_str, interactive)
    return fig


def plot_categorical_spatial(quality_data, month, dekad,
                             quality_prefix="qualitysd", output_dir=None,
                             interactive=True):
    """Categorical quality maps (Poor / Fair / Good / Excellent)."""
    month_str, dekad_str = _format_period(month, dekad)
    n = len(quality_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(
        1, n, figsize=(5.5 * n, 3.5),
        squeeze=False, constrained_layout=True,
    )
    axes = axes.flatten()
    im = None

    for idx, (name, ds) in enumerate(quality_data.items()):
        ax = axes[idx]
        cat = _squeeze_time(ds["categorical_quality"])
        cat_masked = cat.where(cat > 0)
        im = ax.pcolormesh(
            cat_masked.lon, cat_masked.lat, cat_masked.values,
            cmap=_CAT_CMAP, norm=_CAT_NORM, shading="auto",
        )
        ax.set_xlim(*_xlim())
        ax.set_ylim(*_ylim())
        ax.set_title(TITLES_MAP.get(name, name), fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if idx == 0 else "")
        ax.set_aspect("equal")

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes.tolist(), orientation="vertical",
            shrink=0.85, aspect=18, pad=0.02, ticks=[1, 2, 3, 4],
        )
        cbar.ax.set_yticklabels(_CAT_NAMES)
        cbar.set_label("Quality Category")

    fig.suptitle(
        f"Categorical Quality Classification -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )

    _finish_fig(fig, output_dir, "categorical_spatial", quality_prefix,
                month_str, dekad_str, interactive)
    return fig


def plot_improvement(quality_data, month, dekad,
                     quality_prefix="qualitysd", output_dir=None,
                     interactive=True):
    """CQI improvement map (LSEQM+DL minus LS).

    Returns
    -------
    fig : matplotlib.figure.Figure
    stats : dict or None
        Improvement summary (mean, median, pct improved/degraded).
    """
    month_str, dekad_str = _format_period(month, dekad)

    if "LS" not in quality_data or "LSEQMDL" not in quality_data:
        logger.warning("Both LS and LSEQMDL required for improvement map.")
        return None, None

    cqi_ls = _squeeze_time(quality_data["LS"]["continuous_quality"])
    cqi_dl = _squeeze_time(quality_data["LSEQMDL"]["continuous_quality"])
    improvement = cqi_dl - cqi_ls

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    im = ax.pcolormesh(
        improvement.lon, improvement.lat, improvement.values,
        cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto",
    )
    ax.set_xlim(*_xlim())
    ax.set_ylim(*_ylim())
    ax.set_title(
        f"CQI Improvement: LSEQM+DL minus LS -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.colorbar(
        im, ax=ax, orientation="vertical", shrink=0.75, aspect=20,
        pad=0.02, label="CQI Difference (positive = improvement)",
    )

    # Compute summary statistics
    stats = None
    imp_vals = improvement.values[~np.isnan(improvement.values)]
    n_total = len(imp_vals)
    if n_total > 0:
        n_improved = int(np.sum(imp_vals > 0.001))
        n_degraded = int(np.sum(imp_vals < -0.001))
        n_unchanged = n_total - n_improved - n_degraded
        stats = {
            "mean": float(np.mean(imp_vals)),
            "median": float(np.median(imp_vals)),
            "n_total": n_total,
            "n_improved": n_improved,
            "pct_improved": 100.0 * n_improved / n_total,
            "n_degraded": n_degraded,
            "pct_degraded": 100.0 * n_degraded / n_total,
            "n_unchanged": n_unchanged,
            "pct_unchanged": 100.0 * n_unchanged / n_total,
        }
        if interactive:
            print(f"Mean improvement:  {stats['mean']:.4f}")
            print(f"Median improvement: {stats['median']:.4f}")
            print(f"Pixels improved:   {n_improved} "
                  f"({stats['pct_improved']:.1f}%)")
            print(f"Pixels degraded:   {n_degraded} "
                  f"({stats['pct_degraded']:.1f}%)")
            print(f"Pixels unchanged:  {n_unchanged} "
                  f"({stats['pct_unchanged']:.1f}%)")

    _finish_fig(fig, output_dir, "improvement", quality_prefix,
                month_str, dekad_str, interactive)
    return fig, stats


def plot_components(quality_data, month, dekad,
                    quality_prefix="qualitysd", output_dir=None,
                    interactive=True):
    """Component quality maps (basic, distribution, temporal) for best method."""
    month_str, dekad_str = _format_period(month, dekad)

    comp_method = (
        "LSEQMDL" if "LSEQMDL" in quality_data
        else list(quality_data.keys())[-1] if quality_data else None
    )
    if comp_method is None:
        return None

    ds = quality_data[comp_method]
    # Show the three component scores (not overall CQI)
    components = COMPONENT_VARS[:3]

    fig, axes = plt.subplots(
        1, 3, figsize=(16.5, 3.5),
        squeeze=False, constrained_layout=True,
    )
    axes = axes.flatten()
    im = None

    for idx, (var_name, var_label) in enumerate(components):
        ax = axes[idx]
        data = _squeeze_time(ds[var_name])
        im = ax.pcolormesh(
            data.lon, data.lat, data.values,
            cmap="viridis", vmin=0, vmax=1, shading="auto",
        )
        ax.set_xlim(*_xlim())
        ax.set_ylim(*_ylim())
        ax.set_title(var_label, fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if idx == 0 else "")
        ax.set_aspect("equal")

    if im is not None:
        fig.colorbar(
            im, ax=axes.tolist(), orientation="vertical",
            shrink=0.85, aspect=18, pad=0.02,
            label="Score (0 = poor, 1 = excellent)",
        )

    label = TITLES_MAP.get(comp_method, comp_method)
    fig.suptitle(
        f"Quality Components -- {label} -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )

    _finish_fig(fig, output_dir, "components", quality_prefix,
                month_str, dekad_str, interactive)
    return fig


def plot_confidence(quality_data, month, dekad,
                    quality_prefix="qualitysd", output_dir=None,
                    interactive=True):
    """Confidence level map for the best available correction method."""
    month_str, dekad_str = _format_period(month, dekad)

    conf_method = (
        "LSEQMDL" if "LSEQMDL" in quality_data
        else list(quality_data.keys())[-1] if quality_data else None
    )
    if conf_method is None:
        return None

    confidence = _squeeze_time(quality_data[conf_method]["confidence_level"])

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    im = ax.pcolormesh(
        confidence.lon, confidence.lat, confidence.values,
        cmap="YlOrRd_r", vmin=0, vmax=1, shading="auto",
    )
    ax.set_xlim(*_xlim())
    ax.set_ylim(*_ylim())
    label = TITLES_MAP.get(conf_method, conf_method)
    ax.set_title(
        f"Confidence Level -- {label} -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.colorbar(
        im, ax=ax, orientation="vertical", shrink=0.75, aspect=20,
        pad=0.02, label="Confidence (0 = low, 1 = high)",
    )

    if interactive:
        conf_vals = confidence.values[~np.isnan(confidence.values)]
        if len(conf_vals) > 0:
            print(f"Mean confidence:  {np.mean(conf_vals):.4f}")
            print(f"Min confidence:   {np.min(conf_vals):.4f}")
            print(f"Max confidence:   {np.max(conf_vals):.4f}")

    _finish_fig(fig, output_dir, "confidence", quality_prefix,
                month_str, dekad_str, interactive)
    return fig


def plot_cqi_distribution(quality_data, month, dekad,
                          quality_prefix="qualitysd", output_dir=None,
                          interactive=True):
    """Empirical CDF and histogram of CQI across methods."""
    month_str, dekad_str = _format_period(month, dekad)
    if not quality_data:
        return None

    fig, (ax_cdf, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))

    for name, ds in quality_data.items():
        cqi = _squeeze_time(ds["continuous_quality"])
        vals = cqi.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue

        color = METHOD_COLORS.get(name)
        label = TITLES_MAP.get(name, name)

        # Empirical CDF
        sorted_vals = np.sort(vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, ecdf, label=label, color=color,
                    linewidth=1.5)

        # Histogram
        ax_hist.hist(vals, bins=20, alpha=0.45, label=label, color=color,
                     edgecolor="white", linewidth=0.5, range=(0, 1))

        if interactive:
            print(f"Median CQI ({label}): {np.median(vals):.4f}")

    # Category boundary lines
    for thresh in _cat_thresholds():
        ax_cdf.axvline(thresh, color="gray", ls="--", alpha=0.5, lw=0.8)
        ax_hist.axvline(thresh, color="gray", ls="--", alpha=0.5, lw=0.8)

    ax_cdf.set(xlabel="CQI", ylabel="Cumulative Fraction",
               title="Empirical CDF of CQI", xlim=(0, 1), ylim=(0, 1))
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)

    ax_hist.set(xlabel="CQI", ylabel="Pixel Count",
                title="Histogram of CQI", xlim=(0, 1))
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        f"CQI Distribution Analysis -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    _finish_fig(fig, output_dir, "cqi_distribution", quality_prefix,
                month_str, dekad_str, interactive)
    return fig


def plot_category_summary(quality_data, month, dekad,
                          quality_prefix="qualitysd", output_dir=None,
                          interactive=True):
    """Grouped bar chart of quality-category percentages.

    Returns
    -------
    fig : matplotlib.figure.Figure
    summary : dict
        ``{method: {'counts': list, 'total': int, 'pcts': list}}``.
    """
    month_str, dekad_str = _format_period(month, dekad)
    if not quality_data:
        return None, {}

    # Compute counts per category per method
    summary = {}
    for name, ds in quality_data.items():
        cat = _squeeze_time(ds["categorical_quality"])
        vals = cat.values.flatten()
        vals = vals[(vals > 0) & ~np.isnan(vals)]
        total = len(vals)
        counts = [int(np.sum(vals == cv)) for cv in _CAT_VALUES]
        pcts = [100.0 * c / total if total > 0 else 0.0 for c in counts]
        summary[name] = {"counts": counts, "total": total, "pcts": pcts}

    x = np.arange(len(_CAT_NAMES))
    width = 0.25
    n = len(quality_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, info) in enumerate(summary.items()):
        offset = (i - (n - 1) / 2) * width
        label = TITLES_MAP.get(name, name)
        ax.bar(
            x + offset, info["pcts"], width, label=label,
            color=METHOD_COLORS.get(name, f"C{i}"),
            edgecolor="white", linewidth=0.5,
        )

    ax.set(xlabel="Quality Category", ylabel="Percentage of Pixels (%)")
    ax.set_title(
        f"Quality Category Distribution -- Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(_CAT_NAMES)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if interactive:
        hdr = f"{'Method':<12}  {'Poor':>8}  {'Fair':>8}  {'Good':>8}  {'Excellent':>10}  {'Total':>10}"
        print(f"\n{hdr}")
        print("-" * len(hdr))
        for name, info in summary.items():
            label = TITLES_MAP.get(name, name)
            p = info["pcts"]
            print(
                f"{label:<12}  {p[0]:>7.1f}%  {p[1]:>7.1f}%  {p[2]:>7.1f}%"
                f"  {p[3]:>9.1f}%  {info['total']:>10,}"
            )

    _finish_fig(fig, output_dir, "category_summary", quality_prefix,
                month_str, dekad_str, interactive)
    return fig, summary


def plot_component_boxplots(quality_data, month, dekad,
                            quality_prefix="qualitysd", output_dir=None,
                            interactive=True):
    """Box plots of component quality scores across methods."""
    month_str, dekad_str = _format_period(month, dekad)
    if not quality_data:
        return None

    # Try seaborn; fall back to plain matplotlib
    try:
        import seaborn as sns
        _use_sns = True
    except ImportError:
        _use_sns = False

    # Build long-form DataFrame
    records = []
    for name, ds in quality_data.items():
        label = TITLES_MAP.get(name, name)
        for var_name, var_label in COMPONENT_VARS:
            data = _squeeze_time(ds[var_name])
            vals = data.values.flatten()
            vals = vals[~np.isnan(vals)]
            for v in vals:
                records.append(
                    {"Method": label, "Component": var_label, "Score": float(v)}
                )

    df = pd.DataFrame(records)
    if df.empty:
        if interactive:
            print("No data available for box plots.")
        return None

    title = (
        f"Quality Component Scores by Method -- "
        f"Month {month}, Dekad {dekad}"
    )

    if _use_sns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=df, x="Component", y="Score", hue="Method",
            ax=ax, palette="Set2", fliersize=1, linewidth=0.8,
        )
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score (0-1)")
        ax.set_xlabel("Quality Component")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Method")
    else:
        method_order = [TITLES_MAP.get(m, m) for m in quality_data]
        n_comp = len(COMPONENT_VARS)
        fig, axes = plt.subplots(
            1, n_comp, figsize=(4 * n_comp, 6), squeeze=False,
        )
        axes = axes.flatten()
        for c_idx, (_, var_label) in enumerate(COMPONENT_VARS):
            ax = axes[c_idx]
            bp_data = [
                df[(df["Component"] == var_label) & (df["Method"] == m)][
                    "Score"
                ].values
                for m in method_order
            ]
            ax.boxplot(bp_data, labels=method_order, patch_artist=True)
            ax.set_ylim(0, 1)
            ax.set_title(var_label)
            ax.set_ylabel("Score")
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    _finish_fig(fig, output_dir, "component_boxplots", quality_prefix,
                month_str, dekad_str, interactive)
    return fig


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

# Ordered list of all 8 plot functions for batch iteration
_PLOT_FUNCTIONS = [
    ("cqi_spatial", plot_cqi_spatial),
    ("categorical_spatial", plot_categorical_spatial),
    ("improvement", plot_improvement),
    ("components", plot_components),
    ("confidence", plot_confidence),
    ("cqi_distribution", plot_cqi_distribution),
    ("category_summary", plot_category_summary),
    ("component_boxplots", plot_component_boxplots),
]


def run_qa_batch_viz(quality_prefix="qualitysd", output_dir=None,
                     config=None, progress=True):
    """Generate all 8 plot types for all 36 dekadal periods.

    Parameters
    ----------
    quality_prefix : str
        ``'qualitysd'`` or ``'qualityts'``.
    output_dir : str or None
        Base figures directory.  Defaults to
        ``{config.output_dir}/figures/qa``.
    config : module or None
    progress : bool
        Print progress messages.

    Returns
    -------
    dict
        ``{(month, dekad): {'improvement': stats_dict,
        'categories': summary_dict}}``
    """
    if config is None:
        from src import config  # noqa: F811

    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "figures", "qa")

    summary = {}
    n_saved = 0
    n_skipped = 0

    for month, dekad in ALL_DEKADS:
        tag = f"month {month:02d} dekad {dekad}"

        qdata = load_quality_data(
            month, dekad, quality_prefix=quality_prefix, config=config,
        )
        if not qdata:
            n_skipped += 1
            if progress:
                print(f"  {tag} -- skipped (no data)")
            continue

        period_info = {}

        # Call each plot function in batch (non-interactive) mode
        for plot_type, func in _PLOT_FUNCTIONS:
            try:
                result = func(
                    qdata, month, dekad,
                    quality_prefix=quality_prefix,
                    output_dir=output_dir,
                    interactive=False,
                )
                # Collect structured returns where available
                if plot_type == "improvement" and isinstance(result, tuple):
                    _, stats = result
                    period_info["improvement"] = stats
                elif plot_type == "category_summary" and isinstance(result, tuple):
                    _, cat_summary = result
                    period_info["categories"] = cat_summary
            except Exception as exc:
                logger.warning("  %s / %s failed: %s", tag, plot_type, exc)

        # Close datasets to free memory
        for ds in qdata.values():
            ds.close()

        summary[(month, dekad)] = period_info
        n_saved += 1
        if progress:
            n_plots = len(_PLOT_FUNCTIONS)
            print(f"{tag} -- {len(qdata)} method(s), {n_plots} plots")

    if progress:
        print(f"\nBatch complete: {n_saved} periods exported, "
              f"{n_skipped} skipped.")
        print(f"Output directory: {output_dir}")

    return summary


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------

def print_batch_summary(summary):
    """Print aggregated statistics from a batch run.

    Parameters
    ----------
    summary : dict
        Return value of :func:`run_qa_batch_viz`.
    """
    if not summary:
        print("No data to summarise.")
        return

    # Collect improvement stats across periods
    improvements = [
        v["improvement"]
        for v in summary.values()
        if v.get("improvement") is not None
    ]

    if improvements:
        mean_imp = np.mean([s["mean"] for s in improvements])
        mean_pct = np.mean([s["pct_improved"] for s in improvements])
        mean_deg = np.mean([s["pct_degraded"] for s in improvements])
        print("=== Improvement Summary (LSEQM+DL vs LS) ===")
        print(f"  Mean CQI improvement:   {mean_imp:+.4f}")
        print(f"  Avg pixels improved:    {mean_pct:.1f}%")
        print(f"  Avg pixels degraded:    {mean_deg:.1f}%")

        # Best / worst periods
        best = max(improvements, key=lambda s: s["mean"])
        worst = min(improvements, key=lambda s: s["mean"])
        # Find keys
        for key, val in summary.items():
            if val.get("improvement") is best:
                print(f"  Best period:  month {key[0]:02d} dekad {key[1]} "
                      f"(mean +{best['mean']:.4f})")
            if val.get("improvement") is worst:
                print(f"  Worst period: month {key[0]:02d} dekad {key[1]} "
                      f"(mean {worst['mean']:+.4f})")
        print()

    # Collect category distribution across periods
    all_cats = [
        v["categories"]
        for v in summary.values()
        if v.get("categories")
    ]

    if all_cats:
        print("=== Average Category Distribution ===")
        hdr = f"{'Method':<12}  {'Poor':>8}  {'Fair':>8}  {'Good':>8}  {'Excellent':>10}"
        print(hdr)
        print("-" * len(hdr))

        for method in ["LS", "LSEQM", "LSEQMDL"]:
            pcts_all = [
                c[method]["pcts"]
                for c in all_cats
                if method in c
            ]
            if not pcts_all:
                continue
            avg = np.mean(pcts_all, axis=0)
            label = TITLES_MAP.get(method, method)
            print(
                f"{label:<12}  {avg[0]:>7.1f}%  {avg[1]:>7.1f}%  "
                f"{avg[2]:>7.1f}%  {avg[3]:>9.1f}%"
            )
        print()

    print(f"Total periods processed: {len(summary)}")


# ===================================================================
# Station Validation Visualization
# ===================================================================

# ---------------------------------------------------------------------------
# Station validation constants
# ---------------------------------------------------------------------------

# Colours for each product (station validation context)
_STATION_METHOD_COLORS = {
    "IMERG": "#999999",
    "CPC": "#666666",
    "LS": "#e74c3c",
    "LSEQM": "#f39c12",
    "LSEQMDL": "#27ae60",
}

# (variable_name, display_title, colormap, vmin, vmax)
_STATION_METRIC_CONFIGS = [
    ("pearson_correlation", "Pearson Correlation", "RdYlGn", 0, 1),
    ("nse", "Nash-Sutcliffe Efficiency", "RdYlGn", -1, 1),
    ("relative_bias", "Relative Bias", "RdBu_r", -1, 1),
    ("rmse", "RMSE (mm/day)", "YlOrRd", None, None),
    ("csi", "Critical Success Index", "RdYlGn", 0, 1),
    ("pod", "Probability of Detection", "RdYlGn", 0, 1),
]

# WMO multi-threshold verification score panel configs
# (score_key, display_title, y_min, y_max)
_MT_SCORE_CONFIGS = [
    ("pod", "Probability of Detection (POD)", 0, 1),
    ("far", "False Alarm Ratio (FAR)", 0, 1),
    ("csi", "Critical Success Index (CSI)", 0, 1),
    ("fbi", "Frequency Bias Index (FBI)", 0, 3),
    ("ets", "Equitable Threat Score (ETS)", -0.2, 1),
    ("hss", "Heidke Skill Score (HSS)", -0.2, 1),
]


# ---------------------------------------------------------------------------
# Station validation plot functions
# ---------------------------------------------------------------------------

def plot_station_metric_maps(metrics_df, station_df, month, dekad,
                             method_name="LSEQMDL", output_dir=None,
                             interactive=True):
    """Spatial scatter maps of 6 key validation metrics at station locations.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-station metrics (index = WMO station ID).
    station_df : pandas.DataFrame
        Station locations with ``ID_WMO``, ``Lon``, ``Lat`` columns.
    month, dekad : int
        Period identifiers.
    method_name : str
        Correction method label (for the plot title).
    output_dir : str or None
        If given, save to ``{output_dir}/station_validation_metric_maps/``.
    interactive : bool
        ``True`` → ``plt.show()``;  ``False`` → ``plt.close()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    month_str, dekad_str = _format_period(month, dekad)

    if metrics_df.empty:
        logger.warning("plot_station_metric_maps: empty metrics_df, skipping.")
        return None

    # Merge station coordinates (skip columns already present from CSV)
    viz_data = metrics_df.copy()
    needed = [c for c in ["Lon", "Lat"] if c not in viz_data.columns]
    if needed:
        loc_info = station_df[["ID_WMO"] + needed].copy()
        loc_info["ID_WMO"] = loc_info["ID_WMO"].astype(int)
        loc_info = loc_info.set_index("ID_WMO")
        viz_data = viz_data.join(loc_info, how="left")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    display_name = TITLES_MAP.get(method_name, method_name)
    fig.suptitle(
        f"Station Validation Metrics: {display_name}\n"
        f"Month {month}, Dekad {dekad}",
        fontsize=14, fontweight="bold",
    )

    for ax, (metric, title, cmap, vmin, vmax) in zip(
        axes.flat, _STATION_METRIC_CONFIGS
    ):
        if metric not in viz_data.columns:
            ax.set_title(f"{title}\n(not available)")
            continue

        sc = ax.scatter(
            viz_data["Lon"], viz_data["Lat"],
            c=viz_data[metric], cmap=cmap,
            s=35, edgecolors="black", linewidths=0.4,
            vmin=vmin, vmax=vmax, zorder=5,
        )
        ax.set_xlim(*_xlim())
        ax.set_ylim(*_ylim())
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.7)

    plt.tight_layout()
    return _finish_fig(fig, output_dir, "metric_maps",
                       "station_validation", month_str, dekad_str,
                       interactive)


def plot_multi_threshold_curves(all_mt_summaries, month, dekad,
                                output_dir=None, interactive=True):
    """WMO multi-threshold performance curves and exceedance frequency bar chart.

    Parameters
    ----------
    all_mt_summaries : dict of {str: pandas.DataFrame}
        Keyed by method name.  Each DataFrame is indexed by threshold (mm)
        with columns like ``pod_median``, ``pod_p25``, ``pod_p75``, etc.
    month, dekad : int
        Period identifiers.
    output_dir : str or None
        If given, figures are saved into
        ``{output_dir}/station_validation_threshold_curves/``.
    interactive : bool
        ``True`` → ``plt.show()``;  ``False`` → ``plt.close()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Lazy import to avoid circular dependency
    from .station_validation import WMO_THRESHOLDS

    month_str, dekad_str = _format_period(month, dekad)

    if not all_mt_summaries:
        logger.warning("plot_multi_threshold_curves: no summaries, skipping.")
        return None

    # ---- Panel 1: performance curves (2 × 3) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "WMO Multi-Threshold Verification Performance Curves\n"
        f"Month {month}, Dekad {dekad} (WMO/TD-No. 1485)",
        fontsize=14, fontweight="bold",
    )

    for ax, (score, title, ymin, ymax) in zip(axes.flat, _MT_SCORE_CONFIGS):
        for method_name, mt_summary in all_mt_summaries.items():
            med_col = f"{score}_median"
            p25_col = f"{score}_p25"
            p75_col = f"{score}_p75"

            if med_col not in mt_summary.columns:
                continue

            thresholds = mt_summary.index.values
            medians = mt_summary[med_col].values
            color = _STATION_METHOD_COLORS.get(method_name, "#333333")

            ax.plot(thresholds, medians, "o-", color=color,
                    label=method_name, linewidth=2, markersize=5)

            # IQR shading
            if p25_col in mt_summary.columns and p75_col in mt_summary.columns:
                ax.fill_between(
                    thresholds,
                    mt_summary[p25_col].values,
                    mt_summary[p75_col].values,
                    alpha=0.15, color=color,
                )

        # Reference lines
        if score == "fbi":
            ax.axhline(y=1.0, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5, label="Unbiased (FBI=1)")
        if score in ("ets", "hss", "hk"):
            ax.axhline(y=0.0, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5, label="No skill")

        ax.set_xlabel("Threshold (mm/day)")
        ax.set_ylabel(score.upper())
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(list(WMO_THRESHOLDS))
        ax.set_xticklabels([str(t) for t in WMO_THRESHOLDS])
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    _finish_fig(fig, output_dir, "threshold_curves",
                "station_validation", month_str, dekad_str, interactive)

    # ---- Panel 2: exceedance frequency ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle(
        "Exceedance Frequency: Observed vs Product\n"
        f"Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )

    bar_width = 0.12
    x_pos = np.arange(len(WMO_THRESHOLDS))

    for i, (method_name, mt_summary) in enumerate(all_mt_summaries.items()):
        prd_freq = [
            mt_summary.loc[t, "freq_prd_median"]
            if t in mt_summary.index and "freq_prd_median" in mt_summary.columns
            else 0
            for t in WMO_THRESHOLDS
        ]
        color = _STATION_METHOD_COLORS.get(method_name, "#333333")
        ax2.bar(x_pos + i * bar_width, prd_freq, bar_width,
                label=f"{method_name} (product)", color=color, alpha=0.7)

    # Observed frequency (same for all methods — plot once)
    first_summary = next(iter(all_mt_summaries.values()))
    obs_freq = [
        first_summary.loc[t, "freq_obs_median"]
        if t in first_summary.index and "freq_obs_median" in first_summary.columns
        else 0
        for t in WMO_THRESHOLDS
    ]
    n_methods = len(all_mt_summaries)
    ax2.bar(x_pos + n_methods * bar_width, obs_freq, bar_width,
            label="Observed (BMKG)", color="black", alpha=0.5)

    ax2.set_xlabel("Threshold (mm/day)")
    ax2.set_ylabel("Exceedance Frequency (fraction of days)")
    ax2.set_xticks(x_pos + bar_width * n_methods / 2)
    ax2.set_xticklabels([f"{t} mm" for t in WMO_THRESHOLDS])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _finish_fig(fig2, output_dir, "exceedance_frequency",
                "station_validation", month_str, dekad_str, interactive)

    return fig


def plot_regional_boxplots(regional_df, month, dekad,
                           method_name="LSEQMDL", group_col="Region",
                           output_dir=None, interactive=True):
    """Box plots of key validation metrics grouped by region or province.

    Parameters
    ----------
    regional_df : pandas.DataFrame
        Per-station metrics with a ``Region`` or ``Province`` column
        (from ``merge_station_metadata``).
    month, dekad : int
        Period identifiers.
    method_name : str
        Correction method label (for the plot title).
    group_col : str
        Column to group by (``'Region'`` or ``'Province'``).
    output_dir : str or None
        If given, save to ``{output_dir}/station_validation_regional/``.
    interactive : bool
        ``True`` → ``plt.show()``;  ``False`` → ``plt.close()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    month_str, dekad_str = _format_period(month, dekad)

    if regional_df.empty or group_col not in regional_df.columns:
        logger.warning("plot_regional_boxplots: missing '%s' column, "
                       "skipping.", group_col)
        return None

    key_plot_metrics = ["pearson_correlation", "nse", "rmse", "csi"]
    available = [m for m in key_plot_metrics if m in regional_df.columns]
    if not available:
        logger.warning("plot_regional_boxplots: no metrics available.")
        return None

    fig, axes = plt.subplots(1, len(available),
                             figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        # Sort groups by median
        region_order = (
            regional_df.groupby(group_col)[metric]
            .median()
            .sort_values(ascending=False)
            .index
        )
        data_list = [
            regional_df.loc[regional_df[group_col] == r, metric].dropna()
            for r in region_order
        ]
        bp = ax.boxplot(data_list, labels=region_order, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))

        colors = plt.cm.Set3(np.linspace(0, 1, len(region_order)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    display_name = TITLES_MAP.get(method_name, method_name)
    fig.suptitle(
        f"Validation Metrics by {group_col}: {display_name}\n"
        f"Month {month}, Dekad {dekad}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plot_type = f"regional_{group_col.lower()}"
    return _finish_fig(fig, output_dir, plot_type,
                       "station_validation", month_str, dekad_str,
                       interactive)


# ---------------------------------------------------------------------------
# Station validation batch orchestrator
# ---------------------------------------------------------------------------

def run_station_validation_batch_viz(output_dir=None, config=None,
                                     progress=True):
    """Generate station validation plots for all 36 dekadal periods.

    Reads CSV outputs from notebook 05 batch (``station_validation/``)
    and generates three plot types per period:

    1. Station metric scatter maps (6 metrics × spatial)
    2. WMO multi-threshold performance curves + exceedance bar chart
    3. Regional box plots

    Parameters
    ----------
    output_dir : str or None
        Base figures directory.  Defaults to
        ``{config.output_dir}/figures/station_validation``.
    config : module or None
        Configuration module.  If *None*, imports ``src.config``.
    progress : bool
        Print progress messages.

    Returns
    -------
    dict
        ``{(month, dekad): {'n_methods': int, 'n_stations': int, ...}}``
    """
    if config is None:
        from src import config  # noqa: F811

    # Lazy import to avoid circular dependency
    from .station_validation import (
        merge_station_metadata,
        summarize_multi_threshold,
    )
    from .station_density import load_station_locations

    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "figures",
                                  "station_validation")

    # Locate station validation CSV directory
    sv_dir = getattr(config, "STATION_VALIDATION_OUTPUT_DIR", None)
    if sv_dir is None:
        sv_dir = os.path.join(config.output_dir, "station_validation")

    # Load station locations (needed for scatter maps)
    station_df = load_station_locations(config.STATION_FILE)

    summary = {}
    n_saved = 0
    n_skipped = 0

    for month, dekad in ALL_DEKADS:
        month_str, dekad_str = _format_period(month, dekad)
        tag = f"month {month:02d} dekad {dekad}"

        try:
            # --- Load 31-metric CSVs for each method ---
            all_metrics = {}
            for method_key, method_abbr in _METHOD_ABBR.items():
                csv_path = os.path.join(
                    sv_dir,
                    f"station_validation_{method_abbr}"
                    f"_month{month_str}_dekad{dekad_str}.csv",
                )
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, index_col=0)
                    all_metrics[method_key] = df

            if not all_metrics:
                n_skipped += 1
                if progress:
                    print(f"  {tag} -- skipped (no CSVs)")
                continue

            period_info = {"n_methods": len(all_metrics)}

            # --- Plot 1: metric scatter maps (best method) ---
            best = "LSEQMDL" if "LSEQMDL" in all_metrics else list(
                all_metrics.keys())[-1]
            best_df = all_metrics[best]
            period_info["n_stations"] = len(best_df)
            try:
                plot_station_metric_maps(
                    best_df, station_df, month, dekad,
                    method_name=best, output_dir=output_dir,
                    interactive=False,
                )
            except Exception as exc:
                logger.warning("  %s / metric_maps failed: %s", tag, exc)

            # --- Plot 2: multi-threshold curves ---
            all_mt_summaries = {}
            for method_key, method_abbr in _METHOD_ABBR.items():
                mt_csv = os.path.join(
                    sv_dir,
                    f"multi_threshold_summary_{method_abbr}"
                    f"_month{month_str}_dekad{dekad_str}.csv",
                )
                if os.path.exists(mt_csv):
                    mt_df = pd.read_csv(mt_csv, index_col=0)
                    all_mt_summaries[method_key] = mt_df

            if all_mt_summaries:
                try:
                    plot_multi_threshold_curves(
                        all_mt_summaries, month, dekad,
                        output_dir=output_dir, interactive=False,
                    )
                except Exception as exc:
                    logger.warning("  %s / threshold_curves failed: %s",
                                   tag, exc)

            # --- Plot 3: regional box plots ---
            try:
                regional_df = merge_station_metadata(best_df, station_df)
                if "Region" in regional_df.columns:
                    plot_regional_boxplots(
                        regional_df, month, dekad,
                        method_name=best, group_col="Region",
                        output_dir=output_dir, interactive=False,
                    )
            except Exception as exc:
                logger.warning("  %s / regional failed: %s", tag, exc)

            summary[(month, dekad)] = period_info
            n_saved += 1
            if progress:
                print(f"  {tag} -- {len(all_metrics)} method(s), "
                      f"{period_info['n_stations']} stations")

        except Exception as exc:
            n_skipped += 1
            if progress:
                print(f"  {tag} -- FAILED: {exc}")

    if progress:
        print(f"\nStation validation batch complete: {n_saved} periods "
              f"exported, {n_skipped} skipped.")
        print(f"Output directory: {output_dir}")

    return summary


# ---------------------------------------------------------------------------
# Station validation summary reporter
# ---------------------------------------------------------------------------

def print_station_validation_viz_summary(summary):
    """Print aggregated statistics from a station validation batch run.

    Parameters
    ----------
    summary : dict
        Return value of :func:`run_station_validation_batch_viz`.
    """
    if not summary:
        print("No station validation data to summarise.")
        return

    n_stations_all = [
        v["n_stations"] for v in summary.values()
        if v.get("n_stations") is not None
    ]
    n_methods_all = [
        v["n_methods"] for v in summary.values()
        if v.get("n_methods") is not None
    ]

    print("=== Station Validation Visualization Summary ===")
    print(f"  Periods processed:  {len(summary)} / 36")
    if n_stations_all:
        print(f"  Stations per period: "
              f"min {min(n_stations_all)}, "
              f"max {max(n_stations_all)}, "
              f"median {int(np.median(n_stations_all))}")
    if n_methods_all:
        print(f"  Methods per period:  "
              f"min {min(n_methods_all)}, "
              f"max {max(n_methods_all)}")
    print()


# ---------------------------------------------------------------------------
# QA Regional Breakdown  (station-point extraction)
# ---------------------------------------------------------------------------

def _extract_qa_from_datasets(quality_data, station_df):
    """Extract QA variable values from loaded Datasets at station locations.

    Works directly with the ``xr.Dataset`` objects already in memory (the
    return of :func:`load_quality_data`), avoiding re-opening files.

    Parameters
    ----------
    quality_data : dict of {str: xr.Dataset}
        Keyed by display name (``'LS'``, ``'LSEQM'``, ``'LSEQMDL'``).
    station_df : pandas.DataFrame
        Station metadata with ``'ID_WMO'``, ``'Lon'``, ``'Lat'`` columns.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        ``{method_name: DataFrame}`` where each DataFrame has station IDs
        as index and QA variable names as columns.
    """
    result = {}
    for method_name, ds in quality_data.items():
        rows = {}
        var_names = list(ds.data_vars)
        for _, row in station_df.iterrows():
            wmo_id = int(row["ID_WMO"])
            lat = float(row["Lat"])
            lon = float(row["Lon"])
            row_dict = {}
            for var in var_names:
                val = ds[var].sel(lat=lat, lon=lon, method="nearest")
                raw = val.values
                if raw.size == 1:
                    row_dict[var] = raw.item()
                elif raw.size > 1:
                    # qualityts files have a time dimension (per-year);
                    # average across years to get a single representative value
                    finite = raw[np.isfinite(raw)]
                    row_dict[var] = (
                        float(np.nanmean(finite)) if len(finite) > 0
                        else np.nan
                    )
                else:
                    row_dict[var] = np.nan
            rows[wmo_id] = row_dict
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "station_id"
        result[method_name] = df
    return result


def plot_qa_regional_bars(quality_data, station_df, month, dekad,
                          quality_prefix="qualitysd",
                          output_dir=None, interactive=True):
    """Grouped bar chart of median CQI per region for each correction method.

    X-axis shows the 7 island regions (from ``config.ISLAND_ORDER``),
    with one bar per method (LS, LSEQM, LSEQM+DL). Station count is
    annotated above each group.

    Parameters
    ----------
    quality_data : dict of {str: xr.Dataset}
        Return of :func:`load_quality_data`.
    station_df : pandas.DataFrame
        Station metadata with ``'ID_WMO'``, ``'Lon'``, ``'Lat'`` columns.
    month, dekad : int
        Period identifiers.
    quality_prefix : str
        ``'qualitysd'`` or ``'qualityts'``.
    output_dir : str or None
        If given, save into ``{output_dir}/{quality_prefix}_qa_regional/``.
    interactive : bool
        ``True`` → ``plt.show()``;  ``False`` → ``plt.close()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    from .station_validation import merge_station_metadata
    from src import config as _cfg

    month_str, dekad_str = _format_period(month, dekad)

    # Extract QA values at station points
    qa_at_stations = _extract_qa_from_datasets(quality_data, station_df)
    if not qa_at_stations:
        logger.warning("plot_qa_regional_bars: no QA data extracted.")
        return None

    # CQI variable name
    cqi_var = "continuous_quality"

    # Merge metadata and compute medians per region × method
    island_order = _cfg.ISLAND_ORDER
    methods = list(qa_at_stations.keys())
    n_methods = len(methods)
    n_regions = len(island_order)

    medians = np.full((n_methods, n_regions), np.nan)
    counts = np.zeros(n_regions, dtype=int)

    for i, method in enumerate(methods):
        df = qa_at_stations[method]
        merged = merge_station_metadata(df, station_df)
        if "Region" not in merged.columns:
            logger.warning("plot_qa_regional_bars: Region column not "
                           "available after merge.")
            return None
        for j, region in enumerate(island_order):
            vals = merged.loc[merged["Region"] == region, cqi_var].dropna()
            medians[i, j] = vals.median() if len(vals) > 0 else np.nan
            if i == 0:
                counts[j] = len(vals)

    # Plot
    x = np.arange(n_regions)
    width = 0.7 / n_methods
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * width
        display = TITLES_MAP.get(method, method)
        color = METHOD_COLORS.get(method, "#333333")
        ax.bar(x + offset, medians[i], width, label=display, color=color,
               edgecolor="white", linewidth=0.5)

    # Annotate station counts above each group
    for j in range(n_regions):
        y_max = np.nanmax(medians[:, j]) if not np.all(
            np.isnan(medians[:, j])) else 0
        ax.text(x[j], y_max + 0.02, f"n={counts[j]}", ha="center",
                fontsize=8, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(island_order, rotation=30, ha="right")
    ax.set_ylabel("Median CQI")
    ax.set_title(
        f"QA Continuous Quality Index by Region\n"
        f"Month {month}, Dekad {dekad} ({quality_prefix})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    return _finish_fig(fig, output_dir, "qa_regional",
                       quality_prefix, month_str, dekad_str, interactive)


def plot_qa_component_by_region(quality_data, station_df, month, dekad,
                                quality_prefix="qualitysd",
                                output_dir=None, interactive=True):
    """Box plots of QA component scores at stations, grouped by region.

    Four subplots (CQI, basic statistical, distribution, temporal) show
    the distribution of per-station QA values across the 7 island regions.
    Only the best method (LSEQM+DL) is plotted for clarity.

    Parameters
    ----------
    quality_data : dict of {str: xr.Dataset}
    station_df : pandas.DataFrame
    month, dekad : int
    quality_prefix : str
    output_dir : str or None
    interactive : bool

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    from .station_validation import merge_station_metadata
    from src import config as _cfg

    month_str, dekad_str = _format_period(month, dekad)

    # Extract QA at stations
    qa_at_stations = _extract_qa_from_datasets(quality_data, station_df)

    # Use best method
    best = "LSEQMDL" if "LSEQMDL" in qa_at_stations else (
        list(qa_at_stations.keys())[-1] if qa_at_stations else None)
    if best is None:
        return None

    df = qa_at_stations[best]
    merged = merge_station_metadata(df, station_df)
    if "Region" not in merged.columns:
        logger.warning("plot_qa_component_by_region: Region column missing.")
        return None

    island_order = _cfg.ISLAND_ORDER
    # Filter to only components present in the data
    components = [(var, label) for var, label in COMPONENT_VARS
                  if var in merged.columns]
    if not components:
        logger.warning("plot_qa_component_by_region: no component vars found.")
        return None

    n_comp = len(components)
    fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 6),
                             squeeze=False, constrained_layout=True)
    axes = axes[0]

    for ax, (var, label) in zip(axes, components):
        data_list = []
        labels = []
        for region in island_order:
            vals = merged.loc[merged["Region"] == region, var].dropna()
            if len(vals) > 0:
                data_list.append(vals.values)
                labels.append(region)

        if not data_list:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)

        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

    display = TITLES_MAP.get(best, best)
    fig.suptitle(
        f"QA Components by Region: {display}\n"
        f"Month {month}, Dekad {dekad} ({quality_prefix})",
        fontsize=13, fontweight="bold",
    )

    return _finish_fig(fig, output_dir, "qa_component_region",
                       quality_prefix, month_str, dekad_str, interactive)


def plot_qa_province_bars(quality_data, station_df, month, dekad,
                          quality_prefix="qualitysd",
                          output_dir=None, interactive=True):
    """Horizontal bar chart of median CQI per province (best method).

    Provinces are sorted by median CQI (descending). Station count and
    parent region are annotated beside each bar.

    Parameters
    ----------
    quality_data : dict of {str: xr.Dataset}
    station_df : pandas.DataFrame
    month, dekad : int
    quality_prefix : str
    output_dir : str or None
    interactive : bool

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    from .station_validation import merge_station_metadata
    from src import config as _cfg

    month_str, dekad_str = _format_period(month, dekad)

    qa_at_stations = _extract_qa_from_datasets(quality_data, station_df)
    best = "LSEQMDL" if "LSEQMDL" in qa_at_stations else (
        list(qa_at_stations.keys())[-1] if qa_at_stations else None)
    if best is None:
        return None

    cqi_var = "continuous_quality"
    df = qa_at_stations[best]
    merged = merge_station_metadata(df, station_df)
    if "Province" not in merged.columns or cqi_var not in merged.columns:
        logger.warning("plot_qa_province_bars: Province or CQI column "
                       "missing.")
        return None

    # Compute median CQI per province
    prov_stats = (
        merged.groupby("Province")[cqi_var]
        .agg(["median", "count"])
        .rename(columns={"median": "cqi_median", "count": "n"})
        .sort_values("cqi_median", ascending=True)
    )
    prov_stats = prov_stats[prov_stats["n"] > 0]
    if prov_stats.empty:
        return None

    # Map provinces to regions for colour coding
    region_map = _cfg.REGION_MAPPING if hasattr(_cfg, "REGION_MAPPING") else {}
    island_order = _cfg.ISLAND_ORDER
    region_colors = dict(
        zip(island_order,
            plt.cm.Set2(np.linspace(0, 1, len(island_order))))
    )

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(prov_stats) * 0.35)),
        constrained_layout=True,
    )
    y_pos = np.arange(len(prov_stats))
    bar_colors = []
    for prov in prov_stats.index:
        reg = region_map.get(prov, "Other")
        bar_colors.append(region_colors.get(reg, "#bbbbbb"))

    ax.barh(y_pos, prov_stats["cqi_median"], color=bar_colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prov_stats.index, fontsize=9)
    ax.set_xlabel("Median CQI")
    ax.set_xlim(0, 1.1)

    # Annotate counts
    for i, (prov, row) in enumerate(prov_stats.iterrows()):
        ax.text(row["cqi_median"] + 0.01, i, f"n={int(row['n'])}",
                va="center", fontsize=7, color="#555555")

    display = TITLES_MAP.get(best, best)
    ax.set_title(
        f"QA CQI by Province: {display}\n"
        f"Month {month}, Dekad {dekad} ({quality_prefix})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    return _finish_fig(fig, output_dir, "qa_province",
                       quality_prefix, month_str, dekad_str, interactive)


def plot_qa_station_bars(quality_data, station_df, month, dekad,
                         quality_prefix="qualitysd",
                         region_filter=None, output_dir=None,
                         interactive=True):
    """Horizontal bar chart of CQI per individual station (best method).

    When *region_filter* is given, only stations in that region are plotted
    and a per-region figure is saved. Otherwise all stations are included
    in one (possibly large) figure.

    Filenames include ``ID_WMO`` identifiers; province names have spaces
    replaced with underscores for filesystem safety.

    Parameters
    ----------
    quality_data : dict of {str: xr.Dataset}
    station_df : pandas.DataFrame
    month, dekad : int
    quality_prefix : str
    region_filter : str or None
        If set, only plot stations from this region.
    output_dir : str or None
    interactive : bool

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    from .station_validation import merge_station_metadata

    month_str, dekad_str = _format_period(month, dekad)

    qa_at_stations = _extract_qa_from_datasets(quality_data, station_df)
    best = "LSEQMDL" if "LSEQMDL" in qa_at_stations else (
        list(qa_at_stations.keys())[-1] if qa_at_stations else None)
    if best is None:
        return None

    cqi_var = "continuous_quality"
    df = qa_at_stations[best]
    merged = merge_station_metadata(df, station_df)
    if cqi_var not in merged.columns:
        logger.warning("plot_qa_station_bars: CQI column missing.")
        return None

    if region_filter and "Region" in merged.columns:
        merged = merged[merged["Region"] == region_filter]

    if merged.empty:
        return None

    # Build labels: "Station (WMO_ID)"
    if "Station" in merged.columns:
        labels = [f"{row.get('Station', '')} ({idx})"
                  for idx, row in merged.iterrows()]
    else:
        labels = [str(idx) for idx in merged.index]

    sorted_df = merged[[cqi_var]].copy()
    sorted_df["label"] = labels
    sorted_df = sorted_df.sort_values(cqi_var, ascending=True)

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(sorted_df) * 0.28)),
        constrained_layout=True,
    )
    y_pos = np.arange(len(sorted_df))

    # Color by CQI level
    thresholds = _cat_thresholds()  # [poor, fair, good]
    bar_colors = []
    for val in sorted_df[cqi_var]:
        if np.isnan(val) or val < thresholds[0]:
            bar_colors.append(_CAT_COLORS[0])  # Poor
        elif val < thresholds[1]:
            bar_colors.append(_CAT_COLORS[1])  # Fair
        elif val < thresholds[2]:
            bar_colors.append(_CAT_COLORS[2])  # Good
        else:
            bar_colors.append(_CAT_COLORS[3])  # Excellent

    ax.barh(y_pos, sorted_df[cqi_var], color=bar_colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["label"], fontsize=7)
    ax.set_xlabel("CQI")
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="x")

    display = TITLES_MAP.get(best, best)
    region_label = f" — {region_filter}" if region_filter else ""
    ax.set_title(
        f"QA CQI per Station: {display}{region_label}\n"
        f"Month {month}, Dekad {dekad} ({quality_prefix})",
        fontsize=13, fontweight="bold",
    )

    # Use region in filename (spaces removed) for per-region output
    if region_filter:
        region_slug = region_filter.replace(" ", "")
        plot_type = f"qa_station_{region_slug}"
    else:
        plot_type = "qa_station_all"

    return _finish_fig(fig, output_dir, plot_type,
                       quality_prefix, month_str, dekad_str, interactive)


def run_qa_regional_batch_viz(quality_prefix="qualitysd", output_dir=None,
                              config=None, progress=True):
    """Batch: QA regional/province/station plots for all 36 periods.

    For each dekadal period, generates:

    1. Regional grouped bar chart (CQI × method × 7 regions)
    2. Component box plots by region (best method)
    3. Province horizontal bars (best method)
    4. Per-station bars for each region (best method)

    Parameters
    ----------
    quality_prefix : str
        ``'qualitysd'`` or ``'qualityts'``.
    output_dir : str or None
        Base figures directory.  Defaults to
        ``{config.output_dir}/figures/qa``.
    config : module or None
    progress : bool
        Print progress messages.

    Returns
    -------
    dict
        ``{(month, dekad): {'n_stations': int, 'n_methods': int}}``
    """
    if config is None:
        from src import config  # noqa: F811

    from .station_density import load_station_locations

    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "figures", "qa")

    station_df = load_station_locations(config.STATION_FILE)
    island_order = config.ISLAND_ORDER

    summary = {}
    n_saved = 0
    n_skipped = 0

    for month, dekad in ALL_DEKADS:
        tag = f"month {month:02d} dekad {dekad}"

        qdata = load_quality_data(
            month, dekad, quality_prefix=quality_prefix, config=config,
        )
        if not qdata:
            n_skipped += 1
            if progress:
                print(f"  {tag} -- skipped (no data)")
            continue

        period_info = {"n_methods": len(qdata)}

        # 1. Regional CQI bars
        try:
            plot_qa_regional_bars(
                qdata, station_df, month, dekad,
                quality_prefix=quality_prefix,
                output_dir=output_dir, interactive=False,
            )
        except Exception as exc:
            logger.warning("  %s / qa_regional failed: %s", tag, exc)

        # 2. Component box plots by region
        try:
            plot_qa_component_by_region(
                qdata, station_df, month, dekad,
                quality_prefix=quality_prefix,
                output_dir=output_dir, interactive=False,
            )
        except Exception as exc:
            logger.warning("  %s / qa_component_region failed: %s", tag, exc)

        # 3. Province bars
        try:
            plot_qa_province_bars(
                qdata, station_df, month, dekad,
                quality_prefix=quality_prefix,
                output_dir=output_dir, interactive=False,
            )
        except Exception as exc:
            logger.warning("  %s / qa_province failed: %s", tag, exc)

        # 4. Per-station bars (one figure per region)
        n_stations_total = 0
        for region in island_order:
            try:
                plot_qa_station_bars(
                    qdata, station_df, month, dekad,
                    quality_prefix=quality_prefix,
                    region_filter=region,
                    output_dir=output_dir, interactive=False,
                )
            except Exception as exc:
                logger.warning("  %s / qa_station_%s failed: %s",
                               tag, region, exc)

        # Count stations from any method
        qa_ext = _extract_qa_from_datasets(qdata, station_df)
        if qa_ext:
            n_stations_total = len(next(iter(qa_ext.values())))
        period_info["n_stations"] = n_stations_total

        # Close datasets
        for ds in qdata.values():
            ds.close()

        summary[(month, dekad)] = period_info
        n_saved += 1
        if progress:
            print(f"  {tag} -- {len(qdata)} method(s), "
                  f"{n_stations_total} stations, 4 plot types")

    if progress:
        print(f"\nQA regional batch complete: {n_saved} periods exported, "
              f"{n_skipped} skipped.")
        print(f"Output directory: {output_dir}")

    return summary
