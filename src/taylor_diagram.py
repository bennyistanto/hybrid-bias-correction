"""
Module: taylor_diagram.py

This module generates Taylor diagrams [Taylor, 2001] for evaluating
bias-corrected precipitation products against independent BMKG weather
station observations across Indonesia.

Taylor diagrams simultaneously display three complementary statistics
on a single polar plot: Pearson correlation (angular position), normalized
standard deviation (radial distance), and centered RMSE (arc contours).
A product that perfectly reproduces observed variability coincides with
the reference point at correlation = 1.0 and normalized std = 1.0.

The module supports four spatial aggregation levels:
  1. Domain-wide -- all stations pooled into a single diagram.
  2. By island group -- seven main Indonesian island regions.
  3. By province -- one diagram per province (minimum station threshold).
  4. By station -- individual station markers with aggregate overlay.

Products compared: CPC-UNI, IMERG-L, IMERG-F, LS, LSEQM, LSEQM+DL.
Reference: BMKG weather station daily observations (180 stations).

The compute_all_taylor_stats function loops over all 36 dekadal periods,
extracts gridded values at station locations, and accumulates running
statistics via the _Accumulator class. This avoids loading all data
into memory simultaneously. The generate_* functions then produce the
polar plots from the accumulated statistics.

**Author**:
  Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
import os
import warnings
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

PRODUCT_STYLES = OrderedDict([
    ('cpc', {
        'label': 'CPC-UNI', 'marker': 'p',
        'color': '#8c564b', 'size': 11, 'zorder': 5,
    }),
    ('imergl', {
        'label': 'IMERG-L', 'marker': 'o',
        'color': '#1f77b4', 'size': 10, 'zorder': 4,
    }),
    ('imergf', {
        'label': 'IMERG-F', 'marker': 'D',
        'color': '#17becf', 'size': 10, 'zorder': 4,
    }),
    ('ls', {
        'label': 'LS', 'marker': '^',
        'color': '#2ca02c', 'size': 11, 'zorder': 6,
    }),
    ('lseqm', {
        'label': 'LSEQM', 'marker': 's',
        'color': '#ff7f0e', 'size': 10, 'zorder': 7,
    }),
    ('lseqmdl', {
        'label': 'LSEQM+DL', 'marker': '*',
        'color': '#d62728', 'size': 14, 'zorder': 8,
    }),
])

# All 36 dekads: (month, dekad_start_day)
DEKADS = [(m, d) for m in range(1, 13) for d in [1, 11, 21]]
DEKAD_ENDS = {1: 10, 11: 20, 21: 31}

# Province-to-region mapping, island ordering, missing sentinel, and
# minimum valid pairs are loaded from config at call time.
# See config.yml → general, aoi, region_mapping, station_validation sections.


# ──────────────────────────────────────────────────────────────────────
# Running statistics accumulator
# ──────────────────────────────────────────────────────────────────────

class _Accumulator:
    """Numerically stable running accumulator for Taylor diagram statistics.

    Stores per-station, per-product running sums so that statistics
    can be computed for arbitrary spatial groupings by summing the
    station-level accumulators.

    Parameters
    ----------
    product_keys : list of str
        Product identifiers (e.g., ``['cpc', 'imergl', 'ls', ...]``).
    n_stations : int
        Number of stations.
    """

    def __init__(self, product_keys, n_stations):
        n_p = len(product_keys)
        self.product_keys = list(product_keys)
        self.n_stations = n_stations
        self.n = np.zeros((n_p, n_stations), dtype=np.int64)
        self.sum_obs = np.zeros((n_p, n_stations), dtype=np.float64)
        self.sum_prd = np.zeros((n_p, n_stations), dtype=np.float64)
        self.sum_obs2 = np.zeros((n_p, n_stations), dtype=np.float64)
        self.sum_prd2 = np.zeros((n_p, n_stations), dtype=np.float64)
        self.sum_obs_prd = np.zeros((n_p, n_stations), dtype=np.float64)

    def add(self, product_idx, station_idx, obs, prd):
        """Add paired data for one product, one station, one dekad.

        Invalid pairs (NaN, negative) are silently dropped.
        """
        valid = (
            np.isfinite(obs) & np.isfinite(prd)
            & (obs >= 0) & (prd >= 0)
        )
        o = obs[valid].astype(np.float64)
        p = prd[valid].astype(np.float64)
        if len(o) == 0:
            return
        self.n[product_idx, station_idx] += len(o)
        self.sum_obs[product_idx, station_idx] += o.sum()
        self.sum_prd[product_idx, station_idx] += p.sum()
        self.sum_obs2[product_idx, station_idx] += (o * o).sum()
        self.sum_prd2[product_idx, station_idx] += (p * p).sum()
        self.sum_obs_prd[product_idx, station_idx] += (o * p).sum()

    def _compute_from_sums(self, n, s_o, s_p, s_o2, s_p2, s_op, min_n):
        """Compute statistics from accumulated sums (vectorised)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_o = np.where(n > 0, s_o / n, np.nan)
            mean_p = np.where(n > 0, s_p / n, np.nan)
            var_o = np.maximum(s_o2 / np.where(n > 0, n, 1) - mean_o ** 2, 0)
            var_p = np.maximum(s_p2 / np.where(n > 0, n, 1) - mean_p ** 2, 0)
            cov = s_op / np.where(n > 0, n, 1) - mean_o * mean_p

            std_o = np.sqrt(var_o)
            std_p = np.sqrt(var_p)

            denom = std_o * std_p
            corr = np.where(denom > 0, cov / denom, np.nan)
            corr = np.clip(corr, -1.0, 1.0)

            crmse = np.sqrt(np.maximum(var_o + var_p - 2 * cov, 0))
            mse = np.maximum(
                s_p2 / np.where(n > 0, n, 1)
                - 2 * s_op / np.where(n > 0, n, 1)
                + s_o2 / np.where(n > 0, n, 1),
                0,
            )
            rmse = np.sqrt(mse)
            bias = mean_p - mean_o

        # Mask entries with insufficient data
        mask_invalid = n < min_n
        corr = np.where(mask_invalid, np.nan, corr)
        std_o = np.where(mask_invalid, np.nan, std_o)
        std_p = np.where(mask_invalid, np.nan, std_p)
        rmse = np.where(mask_invalid, np.nan, rmse)
        crmse = np.where(mask_invalid, np.nan, crmse)
        bias = np.where(mask_invalid, np.nan, bias)

        return corr, std_o, std_p, rmse, crmse, bias, n

    def compute(self, station_mask=None, min_n=None):
        """Compute Taylor statistics pooled across (masked) stations.

        Parameters
        ----------
        station_mask : array-like of bool, optional
            Pool only stations where ``True``.  If *None*, pool all.
        min_n : int, optional
            Minimum paired observations for a valid result.
            Defaults to ``config.MIN_VALID_DAYS``.

        Returns
        -------
        list of dict
            One dict per product with keys ``key``, ``correlation``,
            ``std_obs``, ``std_prd``, ``rmse``, ``crmse``, ``bias``, ``n``.
        """
        if min_n is None:
            from . import config
            min_n = config.MIN_VALID_DAYS
        if station_mask is not None:
            mask = np.asarray(station_mask, dtype=bool)
            n = self.n[:, mask].sum(axis=1)
            s_o = self.sum_obs[:, mask].sum(axis=1)
            s_p = self.sum_prd[:, mask].sum(axis=1)
            s_o2 = self.sum_obs2[:, mask].sum(axis=1)
            s_p2 = self.sum_prd2[:, mask].sum(axis=1)
            s_op = self.sum_obs_prd[:, mask].sum(axis=1)
        else:
            n = self.n.sum(axis=1)
            s_o = self.sum_obs.sum(axis=1)
            s_p = self.sum_prd.sum(axis=1)
            s_o2 = self.sum_obs2.sum(axis=1)
            s_p2 = self.sum_prd2.sum(axis=1)
            s_op = self.sum_obs_prd.sum(axis=1)

        corr, std_o, std_p, rmse, crmse, bias, n_out = (
            self._compute_from_sums(n, s_o, s_p, s_o2, s_p2, s_op, min_n)
        )

        results = []
        for i, key in enumerate(self.product_keys):
            results.append({
                'key': key,
                'correlation': float(corr[i]),
                'std_obs': float(std_o[i]),
                'std_prd': float(std_p[i]),
                'rmse': float(rmse[i]),
                'crmse': float(crmse[i]),
                'bias': float(bias[i]),
                'n': int(n_out[i]),
            })
        return results

    def compute_per_station(self, min_n=None):
        """Compute per-station Taylor statistics for every product.

        Parameters
        ----------
        min_n : int, optional
            Minimum paired observations.  Defaults to ``config.MIN_VALID_DAYS``.

        Returns
        -------
        dict
            ``{product_key: list_of_station_dicts}``
        """
        if min_n is None:
            from . import config
            min_n = config.MIN_VALID_DAYS
        corr, std_o, std_p, rmse, crmse, bias, n_out = (
            self._compute_from_sums(
                self.n, self.sum_obs, self.sum_prd,
                self.sum_obs2, self.sum_prd2, self.sum_obs_prd, min_n,
            )
        )
        results = {}
        for i, key in enumerate(self.product_keys):
            station_list = []
            for j in range(self.n_stations):
                station_list.append({
                    'correlation': float(corr[i, j]),
                    'std_obs': float(std_o[i, j]),
                    'std_prd': float(std_p[i, j]),
                    'rmse': float(rmse[i, j]),
                    'crmse': float(crmse[i, j]),
                    'bias': float(bias[i, j]),
                    'n': int(n_out[i, j]),
                })
            results[key] = station_list
        return results


# ──────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────

def _subset_for_dekad(ds, var_name, month, dekad_start, dekad_end):
    """Subset a multi-year dataset to a specific dekad."""
    time = ds.time
    mask = (
        (time.dt.month == month)
        & (time.dt.day >= dekad_start)
        & (time.dt.day <= dekad_end)
    )
    subset = ds[var_name].sel(time=mask)
    return subset


def _load_corrected(config, method, month, dekad_start):
    """Load a corrected precipitation file for one dekad.

    Returns *None* if the file does not exist.
    """
    path_dir = getattr(config, f'{method}_corrected_precip_path')
    filename = (
        f'{config.FILENAME_PREFIX}_{method}_corrected_imergl_'
        f'month{month:02d}_dekad{dekad_start:02d}.nc4'
    )
    filepath = os.path.join(path_dir, filename)
    if not os.path.exists(filepath):
        return None
    engine = getattr(config, 'NETCDF_ENGINE', None)
    ds = xr.open_dataset(filepath, engine=engine)
    return ds[config.IMERG_PRECIP_VAR]


def _load_station_data(config):
    """Load station locations and daily observations.

    Returns
    -------
    station_locs : DataFrame
        Columns include ``ID_WMO``, ``Lon``, ``Lat``, and optionally
        ``Province``, ``Region``.
    station_obs : DataFrame
        DatetimeIndex rows, integer WMO-ID columns, values in mm/day.
    """
    station_locs = pd.read_csv(config.STATION_FILE)

    # Ensure Region is available
    if 'Province' in station_locs.columns and 'Region' not in station_locs.columns:
        station_locs['Region'] = station_locs['Province'].map(config.REGION_MAPPING)

    # Locate station observation file
    data_file = getattr(config, 'STATION_DATA_FILE', None)
    if data_file is None:
        # Derive from location file path
        loc_dir = os.path.dirname(config.STATION_FILE)
        data_file = os.path.join(
            loc_dir,
            os.path.basename(config.STATION_FILE).replace('location', 'data'),
        )
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Station data file not found: {data_file}")

    station_obs = pd.read_csv(data_file, index_col=0, parse_dates=True,
                              dayfirst=True)

    # Normalise column names to int (WMO IDs)
    new_cols = {}
    for c in station_obs.columns:
        try:
            new_cols[c] = int(float(c))
        except (ValueError, TypeError):
            new_cols[c] = c
    station_obs = station_obs.rename(columns=new_cols)

    # Replace BMKG missing sentinel
    station_obs = station_obs.replace(config.MISSING_SENTINEL, np.nan)

    return station_locs, station_obs


# ──────────────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────────────

def compute_all_taylor_stats(config=None, progress=True):
    """Compute Taylor statistics for all products at all stations.

    Loops over all 36 dekads, extracts gridded values at BMKG station
    locations, and accumulates running sums for on-the-fly computation
    of correlation, standard deviation, and RMSE.

    Parameters
    ----------
    config : module, optional
        Configuration module (``src.config``).  If *None*, imports it.
    progress : bool
        Print progress messages to stdout.

    Returns
    -------
    acc : _Accumulator
        Per-station running sums for all products.
    station_locs : DataFrame
        Station metadata (``ID_WMO``, ``Lon``, ``Lat``, ``Province``,
        ``Region``).
    """
    if config is None:
        from . import config

    # Load station data
    station_locs, station_obs = _load_station_data(config)
    n_stations = len(station_locs)
    if progress:
        print(f"  Loaded {n_stations} station locations")

    # Determine available products
    product_keys = list(PRODUCT_STYLES.keys())

    # Open large datasets (lazy; no data loaded yet)
    engine = getattr(config, 'NETCDF_ENGINE', None)
    cpc_ds = xr.open_dataset(config.cpc_file, engine=engine)
    imergl_ds = xr.open_dataset(config.imergl_file, engine=engine)

    has_imergf = (
        hasattr(config, 'imergf_file')
        and config.imergf_file
        and os.path.exists(config.imergf_file)
    )
    imergf_ds = (
        xr.open_dataset(config.imergf_file, engine=engine)
        if has_imergf else None
    )
    if not has_imergf:
        product_keys = [k for k in product_keys if k != 'imergf']

    # Initialise accumulator
    acc = _Accumulator(product_keys, n_stations)

    # Station coordinates and IDs
    station_lats = station_locs['Lat'].values
    station_lons = station_locs['Lon'].values
    wmo_ids = station_locs['ID_WMO'].values

    # Map WMO IDs to column lookups in station_obs
    obs_col_map = {}
    for si, wmo in enumerate(wmo_ids):
        wmo_int = int(wmo)
        if wmo_int in station_obs.columns:
            obs_col_map[si] = wmo_int
        elif str(wmo_int) in station_obs.columns:
            obs_col_map[si] = str(wmo_int)

    if progress:
        print(f"  Matched {len(obs_col_map)}/{n_stations} stations to obs data")

    # Process each dekad
    for di, (month, dekad_start) in enumerate(DEKADS):
        dekad_end = DEKAD_ENDS[dekad_start]

        if progress:
            print(
                f"\r  Processing dekad {di + 1:2d}/36 "
                f"(month {month:02d}, days {dekad_start:2d}-{dekad_end:2d}) ...",
                end='', flush=True,
            )

        try:
            # Load gridded products for this dekad
            gridded = {}
            gridded['cpc'] = _subset_for_dekad(
                cpc_ds, config.CPC_PRECIP_VAR, month, dekad_start, dekad_end,
            )
            gridded['imergl'] = _subset_for_dekad(
                imergl_ds, config.IMERG_PRECIP_VAR, month, dekad_start, dekad_end,
            )
            if imergf_ds is not None:
                gridded['imergf'] = _subset_for_dekad(
                    imergf_ds, config.IMERG_PRECIP_VAR,
                    month, dekad_start, dekad_end,
                )

            for method in ['ls', 'lseqm', 'lseqmdl']:
                da = _load_corrected(config, method, month, dekad_start)
                if da is not None:
                    gridded[method] = da

            # Common dates across all loaded products
            date_sets = [
                set(pd.DatetimeIndex(g.time.values)) for g in gridded.values()
            ]
            if not date_sets:
                continue
            common_dates = sorted(set.intersection(*date_sets))

            # Intersect with station observation dates
            obs_index = station_obs.index
            common_dates = pd.DatetimeIndex(
                [d for d in common_dates if d in obs_index]
            )
            if len(common_dates) == 0:
                continue

            # Extract at each station and accumulate
            for si, col in obs_col_map.items():
                obs_ts = station_obs.loc[common_dates, col].values.astype(
                    np.float64
                )
                lat = float(station_lats[si])
                lon = float(station_lons[si])

                for pi, key in enumerate(product_keys):
                    if key not in gridded:
                        continue
                    prd_ts = (
                        gridded[key]
                        .sel(time=common_dates, lat=lat, lon=lon,
                             method='nearest')
                        .values.astype(np.float64)
                    )
                    acc.add(pi, si, obs_ts, prd_ts)

        except Exception as exc:
            if progress:
                print(
                    f"\n  Warning: skipping dekad "
                    f"{month:02d}/{dekad_start:02d}: {exc}"
                )
            continue

    if progress:
        print("\n  Done computing Taylor statistics.")

    # Close datasets
    cpc_ds.close()
    imergl_ds.close()
    if imergf_ds is not None:
        imergf_ds.close()

    return acc, station_locs


# ──────────────────────────────────────────────────────────────────────
# Taylor diagram plotting
# ──────────────────────────────────────────────────────────────────────

def plot_taylor_diagram(
    stats,
    title=None,
    normalize=True,
    figsize=(8, 8),
    ax=None,
    show_rmse_contours=True,
    max_std_ratio=2.0,
    products_to_show=None,
    legend=True,
    rmse_label_angle=None,
):
    """Plot a Taylor diagram from pre-computed statistics.

    Parameters
    ----------
    stats : list of dict
        Each dict must have ``key``, ``correlation``, ``std_obs``,
        ``std_prd``.
    title : str, optional
    normalize : bool
        Divide all standard deviations by the reference std.
    figsize : tuple
        Only used when *ax* is ``None``.
    ax : PolarAxes, optional
        Existing axes to draw on.
    show_rmse_contours : bool
    max_std_ratio : float
        Upper limit for the radial axis (in normalised units if
        ``normalize=True``).
    products_to_show : list of str, optional
        Restrict to these product keys.
    legend : bool
        Add a legend.
    rmse_label_angle : float, optional
        Angle (radians) at which to place RMSE contour labels.  Default
        is a small positive angle.

    Returns
    -------
    fig, ax
    """
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.figure

    # Filter products
    if products_to_show:
        stats = [s for s in stats if s['key'] in products_to_show]
    stats = [s for s in stats if np.isfinite(s.get('correlation', np.nan))]
    if not stats:
        warnings.warn("No valid statistics to plot")
        return fig, ax

    # Reference standard deviation
    ref_std = np.nan
    for s in stats:
        if np.isfinite(s['std_obs']) and s['std_obs'] > 0:
            ref_std = s['std_obs']
            break
    if not np.isfinite(ref_std):
        warnings.warn("Cannot determine reference std; skipping plot")
        return fig, ax

    # ── Axes configuration ──
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    corr_angles = np.degrees(np.arccos(corr_ticks))
    ax.set_thetagrids(corr_angles, labels=[f'{c:.2g}' for c in corr_ticks])

    max_r = max_std_ratio
    ax.set_rlim(0, max_r)
    ax.set_rlabel_position(0)

    # Std-dev axis label
    rlabel = 'Normalized Std Dev' if normalize else 'Std Dev (mm day$^{-1}$)'
    ax.set_ylabel(rlabel, labelpad=30, fontsize=9)

    # ── Reference point ──
    ref_r = 1.0 if normalize else ref_std
    ax.plot(0, ref_r, 'ko', markersize=10, zorder=10, clip_on=False)
    ax.annotate(
        'REF', xy=(0, ref_r),
        xytext=(np.radians(5), ref_r + max_r * 0.04),
        fontsize=8, fontweight='bold', ha='left', va='bottom',
    )

    # ── RMSE contours ──
    if show_rmse_contours:
        if normalize:
            rmse_vals = [v for v in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
                         if v < max_r * 1.5]
        else:
            rmse_vals = [v * ref_std for v in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
                         if v * ref_std < max_r * 1.5]

        label_angle = rmse_label_angle if rmse_label_angle is not None else 0.03

        for ev in rmse_vals:
            phi = np.linspace(0, 2 * np.pi, 400)
            x_cart = ref_r + ev * np.cos(phi)
            y_cart = ev * np.sin(phi)
            r_polar = np.sqrt(x_cart ** 2 + y_cart ** 2)
            theta_polar = np.arctan2(y_cart, x_cart)

            keep = (
                (theta_polar >= -0.01)
                & (theta_polar <= np.pi / 2 + 0.01)
                & (r_polar >= 0)
                & (r_polar <= max_r * 1.02)
            )
            if not keep.any():
                continue

            ax.plot(
                theta_polar[keep], r_polar[keep],
                '--', color='#aaaaaa', linewidth=0.7, alpha=0.6,
            )

            # Label where the arc crosses the zero-correlation axis
            label_r = ref_r + ev
            if label_r <= max_r * 0.98:
                lbl = f'{ev:.2g}' if normalize else f'{ev:.1f}'
                ax.text(
                    label_angle, label_r, lbl,
                    fontsize=7, color='#888888', ha='left', va='bottom',
                )

    # ── Product markers ──
    for s in stats:
        key = s['key']
        style = PRODUCT_STYLES.get(key, {
            'label': key, 'marker': 'o', 'color': 'gray',
            'size': 8, 'zorder': 3,
        })

        theta = np.arccos(np.clip(s['correlation'], 0, 1))
        r = s['std_prd'] / ref_std if normalize else s['std_prd']

        ax.plot(
            theta, r,
            marker=style['marker'], color=style['color'],
            markersize=style['size'], markeredgecolor='black',
            markeredgewidth=0.5, linestyle='none',
            label=style['label'], zorder=style.get('zorder', 5),
            clip_on=False,
        )

    # ── Correlation label along the arc ──
    ax.text(
        np.radians(45), max_r * 1.18, 'Correlation',
        rotation=-45, ha='center', va='center', fontsize=11,
    )

    # ── Title ──
    if title:
        ax.set_title(title, pad=25, fontsize=13, fontweight='bold')

    # ── Legend ──
    if legend:
        ax.legend(
            loc='upper left', bbox_to_anchor=(1.05, 1.0),
            fontsize=9, framealpha=0.9, edgecolor='#cccccc',
        )

    if created_fig:
        fig.tight_layout()

    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Tabular summary
# ──────────────────────────────────────────────────────────────────────

def print_taylor_stats(stats, title=None):
    """Print Taylor statistics as a formatted table.

    Parameters
    ----------
    stats : list of dict
        Output of ``_Accumulator.compute()``.
    title : str, optional
    """
    if title:
        print(f"\n{title}")
        print('=' * len(title))

    header = (
        f"{'Product':<12s} | {'Corr':>6s} | {'σ_obs':>8s} | {'σ_prd':>8s} "
        f"| {'σ ratio':>7s} | {'RMSE':>8s} | {'CRMSE':>8s} | {'Bias':>8s} "
        f"| {'N':>9s}"
    )
    print(header)
    print('-' * len(header))

    for s in stats:
        style = PRODUCT_STYLES.get(s['key'], {'label': s['key']})
        label = style['label']
        r = s['correlation']
        so = s['std_obs']
        sp = s['std_prd']
        ratio = sp / so if np.isfinite(so) and so > 0 else np.nan
        print(
            f"{label:<12s} | {r:6.3f} | {so:8.3f} | {sp:8.3f} "
            f"| {ratio:7.3f} | {s['rmse']:8.3f} | {s['crmse']:8.3f} "
            f"| {s['bias']:8.3f} | {s['n']:9d}"
        )


def save_taylor_stats_csv(acc, station_locs, output_file):
    """Save per-station Taylor statistics to CSV.

    Parameters
    ----------
    acc : _Accumulator
    station_locs : DataFrame
    output_file : str
    """
    per_station = acc.compute_per_station()
    rows = []
    for si in range(acc.n_stations):
        row = {
            'station_id': int(station_locs.iloc[si]['ID_WMO']),
            'station_name': station_locs.iloc[si].get('Station', ''),
            'lon': station_locs.iloc[si]['Lon'],
            'lat': station_locs.iloc[si]['Lat'],
        }
        if 'Province' in station_locs.columns:
            row['province'] = station_locs.iloc[si]['Province']
        if 'Region' in station_locs.columns:
            row['region'] = station_locs.iloc[si]['Region']

        for key in acc.product_keys:
            ss = per_station[key][si]
            row[f'{key}_correlation'] = ss['correlation']
            row[f'{key}_std_obs'] = ss['std_obs']
            row[f'{key}_std_prd'] = ss['std_prd']
            row[f'{key}_rmse'] = ss['rmse']
            row[f'{key}_crmse'] = ss['crmse']
            row[f'{key}_bias'] = ss['bias']
            row[f'{key}_n'] = ss['n']

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"  Saved Taylor statistics CSV: {output_file}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Generation functions
# ──────────────────────────────────────────────────────────────────────

def generate_domain_taylor(acc, station_locs, output_dir=None, **plot_kwargs):
    """Generate and save a domain-wide Taylor diagram.

    Parameters
    ----------
    acc : _Accumulator
    station_locs : DataFrame
    output_dir : str, optional
    **plot_kwargs
        Forwarded to :func:`plot_taylor_diagram`.

    Returns
    -------
    fig : Figure
    """
    stats = acc.compute()
    print_taylor_stats(stats, title='Domain-wide Taylor Statistics')

    fig, ax = plot_taylor_diagram(
        stats,
        title='Domain-wide Taylor Diagram\n(All BMKG Stations)',
        **plot_kwargs,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, 'taylor_domain_wide.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


def generate_island_taylor(acc, station_locs, output_dir=None, **plot_kwargs):
    """Generate per-island Taylor diagrams in a multi-panel figure.

    Parameters
    ----------
    acc : _Accumulator
    station_locs : DataFrame
    output_dir : str, optional
    **plot_kwargs
        Forwarded to :func:`plot_taylor_diagram` (except ``figsize``).

    Returns
    -------
    fig : Figure or None
    """
    regions = station_locs.get('Region')
    if regions is None:
        warnings.warn("No 'Region' column; cannot generate island diagrams")
        return None

    from . import config as _cfg
    island_order = _cfg.ISLAND_ORDER
    available = [isl for isl in island_order if isl in regions.values]
    n_islands = len(available)
    if n_islands == 0:
        warnings.warn("No valid islands found")
        return None

    # Suppress per-panel keywords
    kw = {k: v for k, v in plot_kwargs.items()
           if k not in ('figsize', 'legend')}

    ncols = 3
    nrows = (n_islands + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 6 * nrows),
        subplot_kw={'projection': 'polar'},
    )
    axes = np.atleast_2d(axes)

    for idx, island in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        mask = (regions == island).values
        n_st = int(mask.sum())

        stats = acc.compute(station_mask=mask)
        plot_taylor_diagram(
            stats,
            title=f'{island}\n({n_st} stations)',
            ax=ax,
            legend=False,
            **kw,
        )

    # Hide empty subplots
    for idx in range(n_islands, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared legend from first axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', ncol=len(handles),
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        'Taylor Diagrams by Island Group',
        fontsize=15, fontweight='bold', y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, 'taylor_by_island.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


def generate_province_taylor(
    acc, station_locs, output_dir=None, min_stations=3, **plot_kwargs,
):
    """Generate per-province Taylor diagrams.

    Provinces with fewer than *min_stations* are skipped.

    Parameters
    ----------
    acc : _Accumulator
    station_locs : DataFrame
    output_dir : str, optional
    min_stations : int
        Minimum stations per province.
    **plot_kwargs
        Forwarded to :func:`plot_taylor_diagram`.

    Returns
    -------
    fig : Figure or None
    """
    provinces = station_locs.get('Province')
    if provinces is None:
        warnings.warn("No 'Province' column; cannot generate province diagrams")
        return None

    unique_prov = sorted(provinces.dropna().unique())
    # Filter to provinces with enough stations
    valid_prov = [
        p for p in unique_prov if (provinces == p).sum() >= min_stations
    ]
    n_prov = len(valid_prov)

    if n_prov == 0:
        warnings.warn(f"No provinces with >= {min_stations} stations")
        return None

    kw = {k: v for k, v in plot_kwargs.items()
           if k not in ('figsize', 'legend')}

    ncols = 4
    nrows = (n_prov + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 5.5 * nrows),
        subplot_kw={'projection': 'polar'},
    )
    axes = np.atleast_2d(axes)

    for idx, prov in enumerate(valid_prov):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        mask = (provinces == prov).values
        n_st = int(mask.sum())

        stats = acc.compute(station_mask=mask)
        plot_taylor_diagram(
            stats,
            title=f'{prov}\n({n_st} stations)',
            ax=ax,
            legend=False,
            **kw,
        )

    for idx in range(n_prov, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', ncol=len(handles),
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        'Taylor Diagrams by Province',
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, 'taylor_by_province.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


def generate_station_taylor(
    acc, station_locs, output_dir=None,
    products_to_show=None, normalize=True,
    max_std_ratio=2.0,
):
    """Generate station-level Taylor diagram.

    Individual station markers are shown as small semi-transparent
    points for each product, with domain-wide aggregates overlaid
    as larger opaque markers.

    Parameters
    ----------
    acc : _Accumulator
    station_locs : DataFrame
    output_dir : str, optional
    products_to_show : list of str, optional
    normalize : bool
    max_std_ratio : float

    Returns
    -------
    fig : Figure
    """
    per_station = acc.compute_per_station()
    domain_stats = acc.compute()

    # Reference std from domain aggregate
    ref_std = np.nan
    for s in domain_stats:
        if np.isfinite(s['std_obs']) and s['std_obs'] > 0:
            ref_std = s['std_obs']
            break
    if not np.isfinite(ref_std):
        warnings.warn("Cannot determine reference std")
        return None

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)

    # Axes setup
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ax.set_thetagrids(
        np.degrees(np.arccos(corr_ticks)),
        labels=[f'{c:.2g}' for c in corr_ticks],
    )
    ax.set_rlim(0, max_std_ratio)
    ax.set_rlabel_position(0)

    rlabel = 'Normalized Std Dev' if normalize else 'Std Dev (mm day$^{-1}$)'
    ax.set_ylabel(rlabel, labelpad=30, fontsize=9)

    ref_r = 1.0 if normalize else ref_std

    # Reference point
    ax.plot(0, ref_r, 'ko', markersize=10, zorder=10, clip_on=False)
    ax.annotate(
        'REF', xy=(0, ref_r),
        xytext=(np.radians(5), ref_r + max_std_ratio * 0.04),
        fontsize=8, fontweight='bold',
    )

    # RMSE contours
    for ev in [0.25, 0.5, 0.75, 1.0, 1.25]:
        phi = np.linspace(0, 2 * np.pi, 400)
        x_c = ref_r + ev * np.cos(phi)
        y_c = ev * np.sin(phi)
        r_p = np.sqrt(x_c ** 2 + y_c ** 2)
        t_p = np.arctan2(y_c, x_c)
        keep = (t_p >= 0) & (t_p <= np.pi / 2) & (r_p <= max_std_ratio)
        if keep.any():
            ax.plot(t_p[keep], r_p[keep], '--', color='#aaaaaa',
                    linewidth=0.7, alpha=0.5)

    # ── Per-station markers (small, semi-transparent) ──
    keys_to_plot = products_to_show or acc.product_keys
    for key in keys_to_plot:
        if key not in per_station:
            continue
        style = PRODUCT_STYLES.get(key, {
            'marker': 'o', 'color': 'gray', 'size': 8,
        })
        station_data = per_station[key]

        thetas, rs = [], []
        for ss in station_data:
            if (np.isfinite(ss['correlation'])
                    and np.isfinite(ss['std_prd'])
                    and np.isfinite(ss['std_obs'])
                    and ss['std_obs'] > 0):
                theta = np.arccos(np.clip(ss['correlation'], 0, 1))
                r = ss['std_prd'] / ss['std_obs'] if normalize else ss['std_prd']
                if 0 <= r <= max_std_ratio:
                    thetas.append(theta)
                    rs.append(r)

        if thetas:
            ax.scatter(
                thetas, rs,
                marker=style['marker'], c=style['color'],
                s=18, alpha=0.3, edgecolors='none', zorder=3,
            )

    # ── Domain-wide aggregate markers (large, opaque) ──
    for s in domain_stats:
        if not np.isfinite(s['correlation']):
            continue
        key = s['key']
        if products_to_show and key not in products_to_show:
            continue
        style = PRODUCT_STYLES.get(key, {
            'label': key, 'marker': 'o', 'color': 'gray',
            'size': 10, 'zorder': 5,
        })
        theta = np.arccos(np.clip(s['correlation'], 0, 1))
        r = s['std_prd'] / s['std_obs'] if normalize and s['std_obs'] > 0 else np.nan
        if np.isfinite(r):
            ax.plot(
                theta, r,
                marker=style['marker'], color=style['color'],
                markersize=style['size'] + 2,
                markeredgecolor='black', markeredgewidth=1.0,
                linestyle='none', label=style['label'],
                zorder=style.get('zorder', 5) + 5,
                clip_on=False,
            )

    # Correlation label
    ax.text(
        np.radians(45), max_std_ratio * 1.18, 'Correlation',
        rotation=-45, ha='center', va='center', fontsize=11,
    )

    ax.set_title(
        'Station-Level Taylor Diagram\n'
        '(Individual stations + domain aggregate)',
        pad=25, fontsize=13, fontweight='bold',
    )
    ax.legend(
        loc='upper left', bbox_to_anchor=(1.05, 1.0),
        fontsize=9, framealpha=0.9, edgecolor='#cccccc',
    )
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, 'taylor_by_station.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def generate_all_taylor_diagrams(config=None, output_dir=None):
    """Generate all Taylor diagram variants and save to disk.

    Parameters
    ----------
    config : module, optional
        Configuration module.  If *None*, imports ``src.config``.
    output_dir : str, optional
        Output directory.  Defaults to ``{output_dir}/figures/taylor``.
    """
    if config is None:
        from . import config

    if output_dir is None:
        output_dir = os.path.join(config.output_dir, 'figures', 'taylor')

    print("=" * 60)
    print("  Taylor Diagram Generation")
    print("=" * 60)
    print("\nStep 1/5: Computing statistics across all dekads ...")
    acc, station_locs = compute_all_taylor_stats(config)

    # Save statistics CSV
    print("\nStep 2/5: Saving statistics CSV ...")
    csv_path = os.path.join(output_dir, 'taylor_statistics_per_station.csv')
    save_taylor_stats_csv(acc, station_locs, csv_path)

    print("\nStep 3/5: Domain-wide Taylor diagram ...")
    generate_domain_taylor(acc, station_locs, output_dir)

    print("\nStep 4/5: By-island Taylor diagrams ...")
    generate_island_taylor(acc, station_locs, output_dir)

    print("\nStep 5/5: By-province Taylor diagrams ...")
    generate_province_taylor(acc, station_locs, output_dir)

    # Station-level is optional (large figure)
    print("\nBonus: Station-level Taylor diagram ...")
    generate_station_taylor(acc, station_locs, output_dir)

    print(f"\nAll Taylor diagrams saved to: {output_dir}")
    return acc, station_locs
