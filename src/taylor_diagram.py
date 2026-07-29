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
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.07
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
import os
import gc
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

# Framework-level partition of PRODUCT_STYLES for the split legend
# ("Reference products" vs "Bias-corrected products"). Stable across
# regions because it reflects the correction chain, not local paths.
REFERENCE_KEYS = ('cpc', 'imergl', 'imergf')
TEST_KEYS = ('ls', 'lseqm', 'lseqmdl')

# All 36 dekads: (month, dekad_start_day)
DEKADS = [(m, d) for m in range(1, 13) for d in [1, 11, 21]]
DEKAD_ENDS = {1: 10, 11: 20, 21: 31}

# Province-to-region mapping, island ordering, missing sentinel, and
# minimum valid pairs are loaded from config at call time.
# See config.yml → general, aoi, region_mapping, station_validation sections.


def _product_legend_handles():
    """Build legend handles matching PRODUCT_STYLES for shared figure legends.

    Returns a list of :class:`matplotlib.lines.Line2D` handles suitable
    for ``fig.legend(handles=...)``.
    """
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='REF'),
    ]
    for _key, style in PRODUCT_STYLES.items():
        handles.append(
            Line2D([0], [0], marker=style['marker'], color='w',
                   markerfacecolor=style['color'], markeredgecolor='black',
                   markersize=min(style['size'], 10),
                   label=style['label']),
        )
    return handles


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

    def compute_median(self, station_mask=None, min_n=None):
        """Compute Taylor statistics as median of per-station values.

        Unlike :meth:`compute` which pools raw observations (mixing
        spatial and temporal variance - Simpson's paradox),  this method
        first computes per-station statistics and then takes the median
        across stations.

        Parameters
        ----------
        station_mask : array-like of bool, optional
            Include only stations where ``True``.
        min_n : int, optional
            Minimum paired observations per station.

        Returns
        -------
        list of dict
            Same format as :meth:`compute`.
        """
        per_station = self.compute_per_station(min_n=min_n)
        if station_mask is not None:
            indices = np.where(np.asarray(station_mask, dtype=bool))[0]
        else:
            indices = range(self.n_stations)

        results = []
        for key in self.product_keys:
            station_stats = per_station[key]
            valid = [station_stats[j] for j in indices
                     if np.isfinite(station_stats[j]['correlation'])
                     and np.isfinite(station_stats[j]['std_obs'])
                     and station_stats[j]['std_obs'] > 0
                     and np.isfinite(station_stats[j]['std_prd'])
                     and station_stats[j]['n'] > 0]

            if not valid:
                results.append({
                    'key': key, 'correlation': np.nan,
                    'std_obs': np.nan, 'std_prd': np.nan,
                    'rmse': np.nan, 'crmse': np.nan,
                    'bias': np.nan, 'n': 0,
                })
                continue

            med_corr = float(np.median([s['correlation'] for s in valid]))
            med_std_o = float(np.median([s['std_obs'] for s in valid]))
            med_std_p = float(np.median([s['std_prd'] for s in valid]))
            med_bias = float(np.median([s['bias'] for s in valid]))
            med_crmse = float(np.sqrt(max(
                med_std_o ** 2 + med_std_p ** 2
                - 2 * med_std_o * med_std_p * med_corr, 0,
            )))
            med_rmse = float(np.sqrt(med_crmse ** 2 + med_bias ** 2))

            results.append({
                'key': key,
                'correlation': med_corr,
                'std_obs': med_std_o,
                'std_prd': med_std_p,
                'rmse': med_rmse,
                'crmse': med_crmse,
                'bias': med_bias,
                'n': sum(s['n'] for s in valid),
            })
        return results


class _DekadAggregator:
    """Wraps per-dekad accumulators to compute dekad-level statistics.

    Implements the same interface as :class:`_Accumulator` so that
    ``generate_*`` functions work via duck typing.

    Statistics are computed as: for each dekad, compute per-station
    statistics then take the median across stations; then take the
    median across all 36 dekads.  This evaluates at the bias-correction's
    native temporal resolution rather than mixing seasonal signals.

    Parameters
    ----------
    acc_by_dekad : dict
        Mapping ``{(month, dekad_start): _Accumulator}``.
    """

    def __init__(self, acc_by_dekad):
        self.acc_by_dekad = acc_by_dekad
        first = next(iter(acc_by_dekad.values()))
        self.product_keys = first.product_keys
        self.n_stations = first.n_stations
        # Aggregate n across all dekads (for reporting / min_n checks)
        self.n = sum(a.n for a in acc_by_dekad.values())

    def compute_median(self, station_mask=None, min_n=None):
        """Compute median-of-per-dekad, per-station statistics.

        For each dekad, per-station statistics are computed and then
        the median across stations is taken (one summary per dekad).
        The final result is the median of those 36 dekad summaries.

        Parameters
        ----------
        station_mask : array-like of bool, optional
            Include only stations where ``True``.
        min_n : int, optional
            Minimum paired observations per station per dekad.

        Returns
        -------
        list of dict
            Same format as :meth:`_Accumulator.compute_median`.
        """
        # 1. Per-dekad medians
        per_dekad = [
            acc.compute_median(station_mask=station_mask, min_n=min_n)
            for acc in self.acc_by_dekad.values()
        ]

        # 2. Median across dekads for each product
        results = []
        for pi, key in enumerate(self.product_keys):
            corrs = [d[pi]['correlation'] for d in per_dekad
                     if np.isfinite(d[pi]['correlation'])]
            if not corrs:
                results.append({
                    'key': key, 'correlation': np.nan,
                    'std_obs': np.nan, 'std_prd': np.nan,
                    'rmse': np.nan, 'crmse': np.nan,
                    'bias': np.nan, 'n': 0,
                })
                continue

            std_o = [d[pi]['std_obs'] for d in per_dekad
                     if np.isfinite(d[pi]['std_obs'])]
            std_p = [d[pi]['std_prd'] for d in per_dekad
                     if np.isfinite(d[pi]['std_prd'])]
            biases = [d[pi]['bias'] for d in per_dekad
                      if np.isfinite(d[pi]['bias'])]
            ns = [d[pi]['n'] for d in per_dekad]

            med_corr = float(np.median(corrs))
            med_std_o = float(np.median(std_o)) if std_o else np.nan
            med_std_p = float(np.median(std_p)) if std_p else np.nan
            med_bias = float(np.median(biases)) if biases else 0.0
            med_crmse = float(np.sqrt(max(
                med_std_o ** 2 + med_std_p ** 2
                - 2 * med_std_o * med_std_p * med_corr, 0,
            )))
            med_rmse = float(np.sqrt(med_crmse ** 2 + med_bias ** 2))

            results.append({
                'key': key,
                'correlation': med_corr,
                'std_obs': med_std_o,
                'std_prd': med_std_p,
                'rmse': med_rmse,
                'crmse': med_crmse,
                'bias': med_bias,
                'n': sum(ns),
            })
        return results

    def compute_per_station(self, min_n=None):
        """Compute per-station statistics as median across dekads.

        For each station and product, per-dekad statistics are collected
        and the median across dekads is returned.

        Parameters
        ----------
        min_n : int, optional
            Minimum paired observations per station per dekad.

        Returns
        -------
        dict
            ``{product_key: list_of_station_dicts}``
        """
        # Gather per-dekad per-station stats (list of 36 dicts)
        all_per_station = [
            acc.compute_per_station(min_n=min_n)
            for acc in self.acc_by_dekad.values()
        ]

        results = {}
        for key in self.product_keys:
            station_list = []
            for j in range(self.n_stations):
                corrs, std_os, std_ps = [], [], []
                rmses, crmses, biases, ns = [], [], [], []
                for ps in all_per_station:
                    ss = ps[key][j]
                    if (np.isfinite(ss['correlation'])
                            and np.isfinite(ss['std_obs'])
                            and ss['std_obs'] > 0):
                        corrs.append(ss['correlation'])
                        std_os.append(ss['std_obs'])
                        std_ps.append(ss['std_prd'])
                        rmses.append(ss['rmse'])
                        crmses.append(ss['crmse'])
                        biases.append(ss['bias'])
                        ns.append(ss['n'])

                if not corrs:
                    station_list.append({
                        'correlation': np.nan,
                        'std_obs': np.nan, 'std_prd': np.nan,
                        'rmse': np.nan, 'crmse': np.nan,
                        'bias': np.nan, 'n': 0,
                    })
                else:
                    station_list.append({
                        'correlation': float(np.median(corrs)),
                        'std_obs': float(np.median(std_os)),
                        'std_prd': float(np.median(std_ps)),
                        'rmse': float(np.median(rmses)),
                        'crmse': float(np.median(crmses)),
                        'bias': float(np.median(biases)),
                        'n': sum(ns),
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

    Eagerly loads the DataArray into memory and closes the source file.
    Without this, the parent ``Dataset`` stays open for the lifetime of the
    returned DataArray (xarray lazy loading), and 36 dekads x 3 methods =
    108 file handles + their backing arrays accumulate in
    ``compute_all_taylor_stats`` and OOM Colab.
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
    with xr.open_dataset(filepath, engine=engine) as ds:
        # .load() pulls data into RAM; file is closed when the with-block exits.
        da = ds[config.IMERG_PRECIP_VAR].load()
    return da


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
    station_locs = pd.read_csv(config.STATION_FILE, sep=None, engine='python')

    # Normalize column name variants from station CSV
    rename_map = {}
    for col in station_locs.columns:
        lc = col.lower()
        if lc == 'region' and col != 'Region':
            rename_map[col] = 'Region'
        elif lc in ('a1name', 'province') and col != 'Province':
            rename_map[col] = 'Province'
        elif lc in ('a2name', 'district') and col != 'District':
            rename_map[col] = 'District'
    if rename_map:
        station_locs = station_locs.rename(columns=rename_map)

    # Ensure Region is available (derive from Province if needed)
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

    # Read the CSV and identify the date column.
    # The BMKG station data CSV has columns: ID, Date, JD, <WMO stations...>
    # We need to use 'Date' as the index and drop non-station columns.
    raw = pd.read_csv(data_file)

    # Find and set the date column as index
    date_col = None
    for col in raw.columns:
        if col.lower() == 'date':
            date_col = col
            break

    if date_col is not None:
        raw[date_col] = pd.to_datetime(raw[date_col], format='%d-%m-%Y')
        raw = raw.set_index(date_col)
        # Drop non-station columns (ID, JD, etc.)
        drop_cols = [c for c in raw.columns
                     if c.lower() in ('id', 'jd', 'julian_day', 'doy')]
        raw = raw.drop(columns=drop_cols, errors='ignore')
    else:
        # Fallback: assume first column is date
        try:
            raw = pd.read_csv(data_file, index_col=0, parse_dates=True,
                              date_format='%d-%m-%Y')
        except TypeError:
            raw = pd.read_csv(data_file, index_col=0, parse_dates=True,
                              dayfirst=True)

    station_obs = raw

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

    # Normalise index to date-only (midnight) for reliable date matching
    if isinstance(station_obs.index, pd.DatetimeIndex):
        station_obs.index = station_obs.index.normalize()

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
        Per-station running sums for all products (pooled across dekads).
    station_locs : DataFrame
        Station metadata (``ID_WMO``, ``Lon``, ``Lat``, ``Province``,
        ``Region``).
    acc_by_dekad : dict
        Mapping ``{(month, dekad_start): _Accumulator}`` with separate
        running sums for each of the 36 dekadal periods.
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

    # Initialise accumulators (pooled + per-dekad)
    acc = _Accumulator(product_keys, n_stations)
    acc_by_dekad = {}
    for m, d in DEKADS:
        acc_by_dekad[(m, d)] = _Accumulator(product_keys, n_stations)

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
        if len(station_obs) > 0:
            print(f"  Station obs: {len(station_obs)} rows, "
                  f"{station_obs.index.min()} to {station_obs.index.max()}, "
                  f"dtype={station_obs.index.dtype}")
        else:
            print("  WARNING: station_obs is empty!")

    # Process each dekad - each product paired independently with station obs
    # Normalise to date-only (midnight) so that station DD-MM-YYYY dates
    # match gridded YYYY-MM-DD dates regardless of any time component.
    obs_index = station_obs.index.normalize()
    total_pairs = 0

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

            # Pair each product independently with station observations.
            # Corrected files may have different timestamps than CPC/IMERG
            # (e.g., dummy timestamps from save_corrected_precip), so a
            # global intersection across all products would fail.
            for pi, key in enumerate(product_keys):
                if key not in gridded:
                    continue
                g = gridded[key]

                # Skip products without a time dimension
                if 'time' not in g.dims:
                    logger.debug("  Dekad %02d/%02d: %s has no time dim, "
                                 "skipping", month, dekad_start, key)
                    continue

                # Convert time values to pandas DatetimeIndex, handling
                # both numpy datetime64 and cftime objects.  Normalise
                # to midnight so dates match station obs (no HH:MM:SS).
                raw_times = g.time.values
                try:
                    product_dates = pd.DatetimeIndex(raw_times).normalize()
                except Exception:
                    # cftime objects (e.g., DatetimeGregorian) - 
                    # convert manually to pandas Timestamps
                    try:
                        product_dates = pd.DatetimeIndex([
                            pd.Timestamp(t.year, t.month, t.day)
                            for t in raw_times
                        ])
                    except Exception as cf_exc:
                        logger.warning("  %s: cannot convert times: %s",
                                       key, cf_exc)
                        continue

                # Diagnostic output on first dekad
                if di == 0 and progress:
                    print(f"    {key}: {len(product_dates)} dates, "
                          f"type={type(raw_times[0]).__name__}")

                paired_dates = product_dates.intersection(obs_index)
                if len(paired_dates) == 0:
                    if di == 0 and progress:
                        # Show why pairing failed on first dekad
                        print(f"    {key}: 0 paired dates "
                              f"(product: {product_dates.min()}"
                              f"..{product_dates.max()}, "
                              f"obs: {obs_index.min()}"
                              f"..{obs_index.max()})")
                    continue

                for si, col in obs_col_map.items():
                    obs_ts = station_obs.loc[paired_dates, col].values.astype(
                        np.float64
                    )
                    lat = float(station_lats[si])
                    lon = float(station_lons[si])
                    prd_ts = (
                        g.sel(time=paired_dates, lat=lat, lon=lon,
                              method='nearest')
                        .values.astype(np.float64)
                    )
                    acc.add(pi, si, obs_ts, prd_ts)
                    acc_by_dekad[(month, dekad_start)].add(
                        pi, si, obs_ts, prd_ts)
                    total_pairs += len(paired_dates)

        except Exception as exc:
            if progress:
                print(f"\n    WARN dekad {di + 1}: {exc}")
            logger.warning("  Dekad %02d/%02d failed: %s",
                           month, dekad_start, exc)
            continue
        finally:
            # Release this dekad's gridded arrays before opening the next
            # dekad's corrected files. With 108 corrected NetCDFs across the
            # loop, holding them in memory simultaneously OOMs Colab.
            try:
                gridded.clear()
            except NameError:
                pass
            try:
                del gridded
            except NameError:
                pass
            gc.collect()

    if progress:
        print(f"\n  Done computing Taylor statistics "
              f"({total_pairs:,d} total paired observations).")

    # Close datasets
    cpc_ds.close()
    imergl_ds.close()
    if imergf_ds is not None:
        imergf_ds.close()

    return acc, station_locs, acc_by_dekad


# ──────────────────────────────────────────────────────────────────────
# Taylor diagram plotting (pure matplotlib, no external dependency)
# ──────────────────────────────────────────────────────────────────────

def _draw_taylor_grid(ax, max_std=2.0, compact=False):
    """Draw the Taylor diagram background grid (paper-quality version).

    Layout matches the paper figure 4 helper: bottom x-axis label
    'Standard Deviation' via an annotation, the 'Correlation' label sitting
    at 45 deg close to the outer arc (rotated, green, bold), denser
    correlation ticks with 2-decimal formatting at 0.95+, and CRMSE arcs
    at fixed multiples of std.

    Parameters
    ----------
    ax : matplotlib polar Axes
        Must be created with ``polar=True``.
    max_std : float
        Radial limit (normalised std).
    compact : bool
        Reduced font sizes / fewer correlation labels for multi-panel
        layouts (e.g. monthly 12-panel grid).
    """
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)

    # ── Radial (std) ticks ───────────────────────────────────────
    std_ticks = np.arange(0, max_std + 0.01, 0.5)
    ax.set_rticks(std_ticks)
    ax.set_rlim(0, max_std)
    ax.tick_params(axis='y', labelsize=7 if compact else 8)

    # ── Correlation ticks (denser, 2-decimal at 0.95+) ───────────
    corr_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                  0.7, 0.8, 0.9, 0.95, 0.99]
    if compact:
        show = {0.2, 0.4, 0.6, 0.8, 0.9, 0.99}
        labels = [(f'{c:.2f}' if c >= 0.95 else f'{c:.1f}') if c in show else ''
                  for c in corr_ticks]
        fs_lbl = 7
    else:
        labels = [f'{c:.2f}' if c >= 0.95 else f'{c:.1f}'
                  for c in corr_ticks]
        fs_lbl = 8
    corr_angles = [np.arccos(c) for c in corr_ticks]
    ax.set_thetagrids(np.degrees(corr_angles), labels=labels, fontsize=fs_lbl)

    # ── 'Correlation' label at 45 deg, close to arc ──────────────
    ax.text(np.pi / 4, max_std * (1.06 if compact else 1.07),
            'Corr.' if compact else 'Correlation',
            fontsize=8 if compact else 10,
            color='#2ca02c', fontweight='bold',
            ha='center', va='center',
            rotation=-45, rotation_mode='anchor')

    # ── Radial axis label (left) + bottom x-axis annotation ──────
    ax.set_ylabel('Std Dev' if compact else 'Standard Deviation',
                  fontsize=8 if compact else 10,
                  labelpad=16 if compact else 20)
    ax.annotate('Std Dev' if compact else 'Standard Deviation',
                xy=(0.5, -0.05 if compact else -0.04),
                xycoords='axes fraction',
                ha='center', va='top',
                fontsize=8 if compact else 10)

    # ── Reference point (REF marker) ─────────────────────────────
    ax.plot(0, 1.0, 'ko', markersize=6 if compact else 8, zorder=10)

    # ── CRMSE arcs (dashed, centred on reference) ────────────────
    for crmse in [0.5, 1.0, 1.5]:
        theta = np.linspace(0, np.pi / 2, 200)
        r_arc = []
        for t in theta:
            costh = np.cos(t)
            disc = costh ** 2 - (1 - crmse ** 2)
            r_arc.append(costh + np.sqrt(disc) if disc >= 0 else np.nan)
        r_arc = np.array(r_arc)
        mask = (r_arc >= 0) & (r_arc <= max_std)
        if mask.any():
            ax.plot(theta[mask], r_arc[mask], '--',
                    color='gray',
                    linewidth=0.4 if compact else 0.5,
                    alpha=0.6, zorder=1)
    ax.grid(True, alpha=0.3)


def _legend_handles_split():
    """Return (reference, test) legend handles for the split legend layout.

    Pairs with the paper figure 4 design: two separate ``fig.legend(...)``
    calls - one titled 'Reference products', one 'Bias-corrected products'.
    Selection of which key is which is controlled by REFERENCE_KEYS and
    TEST_KEYS module constants.
    """
    from matplotlib.lines import Line2D

    ref = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='black', markeredgecolor='black',
                  markersize=7, label='REF (station)')]
    for k in REFERENCE_KEYS:
        if k not in PRODUCT_STYLES:
            continue
        s = PRODUCT_STYLES[k]
        ref.append(Line2D([0], [0], marker=s['marker'], color='w',
                          markerfacecolor=s['color'],
                          markeredgecolor='black',
                          markersize=min(s['size'], 10),
                          label=s['label']))
    tst = []
    for k in TEST_KEYS:
        if k not in PRODUCT_STYLES:
            continue
        s = PRODUCT_STYLES[k]
        tst.append(Line2D([0], [0], marker=s['marker'], color='w',
                          markerfacecolor=s['color'],
                          markeredgecolor='black',
                          markersize=min(s['size'], 10),
                          label=s['label']))
    return ref, tst


def plot_taylor_diagram(
    stats,
    title=None,
    normalize=True,
    figsize=(8, 8),
    ax=None,
    max_std_ratio=2.0,
    products_to_show=None,
    legend=True,
    **_ignored,
):
    """Plot a Taylor diagram using PRODUCT_STYLES markers.

    Uses a custom polar grid (no external dependency).  Markers
    match :data:`PRODUCT_STYLES` exactly, so shared legends built
    with :func:`_product_legend_handles` are always consistent.

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
    ax : matplotlib.axes.Axes, optional
        Existing **regular** (non-polar) axes for subplot usage.
        A polar axes is created at the same position automatically.
    max_std_ratio : float
        Upper limit for the radial axis.
    products_to_show : list of str, optional
        Restrict to these product keys.
    legend : bool
        Add a legend (standalone mode only).

    Returns
    -------
    fig, ax
    """
    # Filter products
    if products_to_show:
        stats = [s for s in stats if s['key'] in products_to_show]
    valid = [s for s in stats
             if np.isfinite(s.get('correlation', np.nan))
             and np.isfinite(s.get('std_prd', np.nan))
             and s.get('n', 0) > 0]

    compact = ax is not None

    # ── Create or replace axes ───────────────────────────────────
    if ax is not None:
        fig = ax.figure
        pos = ax.get_position()
        label_id = ax.get_label()
        ax.remove()
        ax = fig.add_axes(pos, polar=True, label=label_id)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

    if not valid:
        warnings.warn("No valid statistics to plot")
        return fig, ax

    # Reference std
    ref_std = np.nan
    for s in valid:
        if np.isfinite(s['std_obs']) and s['std_obs'] > 0:
            ref_std = s['std_obs']
            break
    if not np.isfinite(ref_std):
        warnings.warn("Cannot determine reference std; skipping plot")
        return fig, ax

    # ── Draw background grid ─────────────────────────────────────
    _draw_taylor_grid(ax, max_std=max_std_ratio, compact=compact)

    # ── Plot product markers ─────────────────────────────────────
    for s in valid:
        style = PRODUCT_STYLES.get(s['key'], {
            'label': s['key'], 'marker': 'o',
            'color': 'gray', 'size': 10, 'zorder': 5,
        })
        sp = s['std_prd'] / ref_std if normalize else s['std_prd']
        theta = np.arccos(np.clip(s['correlation'], 0, 1))
        ms = style['size'] if not compact else max(style['size'] - 2, 5)

        ax.plot(
            theta, sp,
            marker=style['marker'], color=style['color'],
            markersize=ms, markeredgecolor='black',
            markeredgewidth=0.5, linestyle='none',
            zorder=style.get('zorder', 5),
            label=style['label'],
        )

    # ── Legend (standalone mode only) ────────────────────────────
    if legend and not compact:
        handles = _product_legend_handles()
        ax.legend(handles=handles, loc='upper right', fontsize=8,
                  framealpha=0.9, markerscale=0.9)

    if title:
        pad = 15 if compact else 25
        fsize = 10 if compact else 13
        ax.set_title(title, pad=pad, fontsize=fsize, fontweight='bold')

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

def _format_diagram_label(diagram_label):
    """Convert a diagram_label to (label_text, file_suffix).

    Returns
    -------
    label_text : str
        Human-readable label (e.g. 'Per-Dekad',
        'Month 01 Dekad 01'), or '' if *diagram_label* is falsy.
    file_suffix : str
        Filename suffix (e.g. '_per_dekad',
        '_month01_dekad01'), or '' if *diagram_label* is falsy.

    Examples
    --------
    >>> _format_diagram_label(None)
    ('', '')
    >>> _format_diagram_label('per_dekad')
    ('Per-Dekad', '_per_dekad')
    >>> _format_diagram_label('month01_dekad01')
    ('Month 01 Dekad 01', '_month01_dekad01')
    """
    if not diagram_label:
        return '', ''
    _LABEL_MAP = {'per_dekad': 'Per-Dekad'}
    if diagram_label in _LABEL_MAP:
        text = _LABEL_MAP[diagram_label]
    elif diagram_label.startswith('month'):
        # e.g. 'month01_dekad01' -> 'Month 01 Dekad 01'
        text = diagram_label.replace('_', ' ').title()
    else:
        text = diagram_label
    return text, f'_{diagram_label}'


def generate_domain_taylor(acc, station_locs, output_dir=None,
                           diagram_label=None, **plot_kwargs):
    """Generate and save a domain-wide Taylor diagram.

    Parameters
    ----------
    acc : _Accumulator or _DekadAggregator
    station_locs : DataFrame
    output_dir : str, optional
    diagram_label : str, optional
        Label suffix for title and filename (e.g. ``'per_dekad'``).
    **plot_kwargs
        Forwarded to :func:`plot_taylor_diagram`.

    Returns
    -------
    fig : Figure
    """
    label_text, file_suffix = _format_diagram_label(diagram_label)
    label_suffix = f' \u2014 {label_text}' if label_text else ''

    stats = acc.compute_median()
    print_taylor_stats(
        stats,
        title=f'Domain-wide Taylor Statistics{label_suffix} (median of per-station)',
    )

    fig, ax = plot_taylor_diagram(
        stats,
        title=f'Domain-wide Taylor Diagram\n(All BMKG Stations{label_suffix})',
        **plot_kwargs,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, f'taylor_domain_wide{file_suffix}.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


def generate_island_taylor(acc, station_locs, output_dir=None,
                           diagram_label=None, **plot_kwargs):
    """Generate per-island Taylor diagrams in a multi-panel figure.

    Parameters
    ----------
    acc : _Accumulator or _DekadAggregator
    station_locs : DataFrame
    output_dir : str, optional
    diagram_label : str, optional
        Label suffix for title and filename (e.g. ``'per_dekad'``).
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
    )
    axes = np.atleast_2d(axes)

    # Track which axes were replaced by polar axes
    active_axes = {}

    for idx, island in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        mask = (regions == island).values
        n_st = int(mask.sum())

        stats = acc.compute_median(station_mask=mask)
        _, new_ax = plot_taylor_diagram(
            stats,
            title=f'{island}\n({n_st} stations)',
            ax=ax,
            legend=False,
            **kw,
        )
        active_axes[(row, col)] = new_ax

    # Hide empty subplots
    for idx in range(n_islands, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Split legend (Reference products vs Bias-corrected products),
    # matching the paper figure 4 layout. Anchored at the bottom of the
    # figure so it does not overlap the subplots.
    ref_h, tst_h = _legend_handles_split()
    fig.legend(handles=ref_h, title='Reference products',
               loc='lower center', bbox_to_anchor=(0.30, -0.04),
               ncol=len(ref_h), fontsize=9, framealpha=0.9,
               title_fontsize=9)
    fig.legend(handles=tst_h, title='Bias-corrected products',
               loc='lower center', bbox_to_anchor=(0.75, -0.04),
               ncol=len(tst_h), fontsize=9, framealpha=0.9,
               title_fontsize=9)

    label_text, file_suffix = _format_diagram_label(diagram_label)
    label_suffix = f' ({label_text})' if label_text else ''

    fig.suptitle(
        f'Taylor Diagrams by Island Group{label_suffix}',
        fontsize=15, fontweight='bold', y=1.01,
    )
    # constrained_layout does not reserve space for fig-level legends;
    # explicit subplots_adjust does.
    fig.subplots_adjust(bottom=0.10, top=0.94, left=0.05, right=0.97,
                        wspace=0.30, hspace=0.40)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, f'taylor_by_island{file_suffix}.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


def generate_province_taylor(
    acc, station_locs, output_dir=None, min_stations=3,
    diagram_label=None, **plot_kwargs,
):
    """Generate per-province Taylor diagrams.

    Provinces with fewer than *min_stations* are skipped.

    Parameters
    ----------
    acc : _Accumulator or _DekadAggregator
    station_locs : DataFrame
    output_dir : str, optional
    min_stations : int
        Minimum stations per province.
    diagram_label : str, optional
        Label suffix for title and filename (e.g. ``'per_dekad'``).
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
    )
    axes = np.atleast_2d(axes)

    # Track which axes were replaced by polar axes
    active_axes = {}

    for idx, prov in enumerate(valid_prov):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        mask = (provinces == prov).values
        n_st = int(mask.sum())

        stats = acc.compute_median(station_mask=mask)
        _, new_ax = plot_taylor_diagram(
            stats,
            title=f'{prov}\n({n_st} stations)',
            ax=ax,
            legend=False,
            **kw,
        )
        active_axes[(row, col)] = new_ax

    for idx in range(n_prov, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Split legend (Reference products vs Bias-corrected products),
    # matching the paper figure 4 layout.
    ref_h, tst_h = _legend_handles_split()
    fig.legend(handles=ref_h, title='Reference products',
               loc='lower center', bbox_to_anchor=(0.30, -0.04),
               ncol=len(ref_h), fontsize=9, framealpha=0.9,
               title_fontsize=9)
    fig.legend(handles=tst_h, title='Bias-corrected products',
               loc='lower center', bbox_to_anchor=(0.75, -0.04),
               ncol=len(tst_h), fontsize=9, framealpha=0.9,
               title_fontsize=9)

    label_text, file_suffix = _format_diagram_label(diagram_label)
    label_suffix = f' ({label_text})' if label_text else ''

    fig.suptitle(
        f'Taylor Diagrams by Province{label_suffix}',
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.subplots_adjust(bottom=0.08, top=0.94, left=0.05, right=0.97,
                        wspace=0.30, hspace=0.40)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, f'taylor_by_province{file_suffix}.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


def generate_station_taylor(
    acc, station_locs, output_dir=None,
    products_to_show=None, normalize=True,
    max_std_ratio=2.0, diagram_label=None,
):
    """Generate station-level Taylor diagram.

    Individual station markers are shown as small semi-transparent
    points for each product, with median-of-per-station aggregates
    overlaid as larger opaque markers.  Each station is normalised
    by its own observation standard deviation so that stations with
    different climatologies contribute comparably.

    Parameters
    ----------
    acc : _Accumulator or _DekadAggregator
    station_locs : DataFrame
    output_dir : str, optional
    products_to_show : list of str, optional
    normalize : bool
    max_std_ratio : float
    diagram_label : str, optional
        Label suffix for title and filename (e.g. ``'per_dekad'``).

    Returns
    -------
    fig : Figure

    Notes
    -----
    Direct pooling (``acc.compute()``) mixes spatial and temporal
    variance: stations with different climates inflate the pooled
    variance and dilute the temporal correlation (Simpson's paradox).
    Individual stations may show corr 0.6--0.95, while the pooled
    correlation drops to ~0.3.  Using the median of per-station
    normalised statistics gives a representative *typical station*
    aggregate that is visually consistent with the scatter cloud.
    """
    per_station = acc.compute_per_station()

    # ── Aggregate as MEDIAN of per-station normalised metrics ────
    agg_stats = []
    for key in acc.product_keys:
        st_data = per_station.get(key, [])
        valid = [s for s in st_data
                 if np.isfinite(s['correlation'])
                 and np.isfinite(s['std_obs']) and s['std_obs'] > 0
                 and np.isfinite(s['std_prd'])
                 and s['n'] > 0]
        if not valid:
            agg_stats.append({
                'key': key, 'correlation': np.nan,
                'std_obs': 1.0, 'std_prd': np.nan,
                'crmse': np.nan, 'rmse': np.nan, 'bias': np.nan, 'n': 0,
            })
            continue

        corrs = [s['correlation'] for s in valid]
        std_ratios = [s['std_prd'] / s['std_obs'] for s in valid]

        med_corr = float(np.median(corrs))
        med_ratio = float(np.median(std_ratios))
        # Self-consistent normalised CRMSE from Taylor identity:
        # CRMSE² = σ_o² + σ_p² − 2 σ_o σ_p r  (with σ_o = 1)
        med_ncrmse = float(np.sqrt(max(
            1.0 + med_ratio ** 2 - 2.0 * med_ratio * med_corr, 0,
        )))

        agg_stats.append({
            'key': key,
            'correlation': med_corr,
            'std_obs': 1.0,           # normalised reference
            'std_prd': med_ratio,     # median σ_prd / σ_obs
            'crmse': med_ncrmse,
            'rmse': np.nan,
            'bias': np.nan,
            'n': sum(s['n'] for s in valid),
        })

    has_valid = any(
        np.isfinite(s.get('correlation', np.nan)) and s.get('n', 0) > 0
        for s in agg_stats
    )
    if not has_valid:
        warnings.warn("No valid per-station statistics")
        return None

    # ── Plot aggregate markers (already in normalised space) ─────
    label_text, file_suffix = _format_diagram_label(diagram_label)
    label_suffix = f' ({label_text})' if label_text else ''

    fig, ax = plot_taylor_diagram(
        agg_stats,
        title=(f'Station-Level Taylor Diagram{label_suffix}\n'
               '(individual stations + median aggregate)'),
        normalize=normalize,
        figsize=(9, 9),
        max_std_ratio=max_std_ratio,
        products_to_show=products_to_show,
        legend=True,
    )

    # ── Overlay per-station scatter ──────────────────────────────
    # Each station normalised by its own σ_obs for consistency
    # with the median-aggregate markers.
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
                # Normalise by station's own σ_obs
                r = (ss['std_prd'] / ss['std_obs']
                     if normalize else ss['std_prd'])
                if 0 <= r <= max_std_ratio:
                    thetas.append(theta)
                    rs.append(r)

        if thetas:
            ax.scatter(
                thetas, rs,
                marker=style['marker'], c=style['color'],
                s=18, alpha=0.3, edgecolors='none', zorder=2,
            )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, f'taylor_by_station{file_suffix}.png')
        fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fp}")

    return fig


# ──────────────────────────────────────────────────────────────────────
# Per-dekad orchestrators
# ──────────────────────────────────────────────────────────────────────

def generate_per_dekad_taylor_diagrams(
    acc_by_dekad, station_locs, output_dir=None,
    individual=True, summary=True,
):
    """Generate Taylor diagrams evaluated at per-dekad resolution.

    Two modes:

    * **Summary** (``summary=True``): one diagram per spatial level
      showing the median across all 36 dekads.  Uses
      :class:`_DekadAggregator`.

    * **Individual** (``individual=True``): one diagram per dekad per
      spatial level, with ``_month{MM}_dekad{DD}`` in the filename.

    Parameters
    ----------
    acc_by_dekad : dict
        Mapping ``{(month, dekad_start): _Accumulator}``.
    station_locs : DataFrame
    output_dir : str, optional
    individual : bool
        Generate one diagram per dekad (36 × spatial-levels).
    summary : bool
        Generate summary diagrams (median across dekads).
    """
    if summary:
        print("\n  Per-dekad summary (median across 36 dekads) ...")
        dagg = _DekadAggregator(acc_by_dekad)

        fig = generate_domain_taylor(
            dagg, station_locs, output_dir,
            diagram_label='per_dekad', normalize=True,
            max_std_ratio=2.0,
        )
        if fig is not None:
            plt.close(fig)

        fig = generate_island_taylor(
            dagg, station_locs, output_dir,
            diagram_label='per_dekad', normalize=True,
        )
        if fig is not None:
            plt.close(fig)

        fig = generate_province_taylor(
            dagg, station_locs, output_dir,
            diagram_label='per_dekad', min_stations=3,
            normalize=True,
        )
        if fig is not None:
            plt.close(fig)

        fig = generate_station_taylor(
            dagg, station_locs, output_dir,
            diagram_label='per_dekad', normalize=True,
            max_std_ratio=2.0,
        )
        if fig is not None:
            plt.close(fig)

    if individual:
        print("\n  Per-dekad individual diagrams (36 periods) ...")
        sorted_keys = sorted(acc_by_dekad.keys())
        for di, (month, dekad_start) in enumerate(sorted_keys):
            acc_dk = acc_by_dekad[(month, dekad_start)]
            tag = f'month{month:02d}_dekad{dekad_start:02d}'
            dk_label = f'month{month:02d}_dekad{dekad_start:02d}'

            print(f"\r    Dekad {di + 1:2d}/36 ({tag}) ...",
                  end='', flush=True)

            # Domain-wide
            try:
                fig = generate_domain_taylor(
                    acc_dk, station_locs, output_dir,
                    diagram_label=dk_label, normalize=True,
                    max_std_ratio=2.0,
                )
                if fig is not None:
                    plt.close(fig)
            except Exception as exc:
                logger.warning("  Dekad %s domain failed: %s", tag, exc)

            # By-island
            try:
                fig = generate_island_taylor(
                    acc_dk, station_locs, output_dir,
                    diagram_label=dk_label, normalize=True,
                )
                if fig is not None:
                    plt.close(fig)
            except Exception as exc:
                logger.warning("  Dekad %s island failed: %s", tag, exc)

            # By-province
            try:
                fig = generate_province_taylor(
                    acc_dk, station_locs, output_dir,
                    diagram_label=dk_label, min_stations=3,
                    normalize=True,
                )
                if fig is not None:
                    plt.close(fig)
            except Exception as exc:
                logger.warning("  Dekad %s province failed: %s", tag, exc)

            # Station-level
            try:
                fig = generate_station_taylor(
                    acc_dk, station_locs, output_dir,
                    diagram_label=dk_label, normalize=True,
                    max_std_ratio=2.0,
                )
                if fig is not None:
                    plt.close(fig)
            except Exception as exc:
                logger.warning("  Dekad %s station failed: %s", tag, exc)

            # End-of-dekad cleanup. plt.close(fig) per figure is not enough
            # on Colab - matplotlib's internal state for polar axes + CRMSE
            # arc collections accumulates until OOM around dekad ~15.
            # plt.close('all') drops any straggling figures; gc.collect()
            # forces release of the C-level memory.
            plt.close('all')
            gc.collect()

        print()  # newline after progress dots


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def save_dekad_taylor_stats_csv(acc_by_dekad, station_locs, output_file):
    """Save per-station per-dekad Taylor statistics to CSV.

    Parameters
    ----------
    acc_by_dekad : dict
        Mapping ``{(month, dekad_start): _Accumulator}``.
    station_locs : DataFrame
    output_file : str

    Returns
    -------
    DataFrame
    """
    rows = []
    for (month, dekad_start), acc in sorted(acc_by_dekad.items()):
        per_station = acc.compute_per_station()
        for si in range(acc.n_stations):
            row = {
                'station_id': int(station_locs.iloc[si]['ID_WMO']),
                'month': month,
                'dekad_start': dekad_start,
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
                row[f'{key}_n'] = ss['n']

            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"  Saved per-dekad Taylor statistics CSV: {output_file}")
    return df


def generate_all_taylor_diagrams(config=None, output_dir=None):
    """Generate all Taylor diagram variants and save to disk.

    Generates both pooled-across-dekads and per-dekad diagrams.

    Parameters
    ----------
    config : module, optional
        Configuration module.  If *None*, imports ``src.config``.
    output_dir : str, optional
        Output directory.  Defaults to ``{output_dir}/figures/taylor``.

    Returns
    -------
    acc : _Accumulator
    station_locs : DataFrame
    acc_by_dekad : dict
    """
    if config is None:
        from . import config

    if output_dir is None:
        output_dir = os.path.join(config.output_dir, 'figures', 'taylor')

    print("=" * 60)
    print("  Taylor Diagram Generation")
    print("=" * 60)
    print("\nStep 1/8: Computing statistics across all dekads ...")
    acc, station_locs, acc_by_dekad = compute_all_taylor_stats(config)

    # Save statistics CSVs
    print("\nStep 2/8: Saving pooled statistics CSV ...")
    csv_path = os.path.join(output_dir, 'taylor_statistics_per_station.csv')
    save_taylor_stats_csv(acc, station_locs, csv_path)

    print("\nStep 3/8: Saving per-dekad statistics CSV ...")
    csv_dekad = os.path.join(output_dir, 'taylor_statistics_per_dekad.csv')
    save_dekad_taylor_stats_csv(acc_by_dekad, station_locs, csv_dekad)

    # Pooled diagrams (close figures to conserve memory in batch)
    print("\nStep 4/8: Domain-wide Taylor diagram (pooled) ...")
    fig = generate_domain_taylor(acc, station_locs, output_dir)
    if fig is not None:
        plt.close(fig)

    print("\nStep 5/8: By-island Taylor diagrams (pooled) ...")
    fig = generate_island_taylor(acc, station_locs, output_dir)
    if fig is not None:
        plt.close(fig)

    print("\nStep 6/8: By-province Taylor diagrams (pooled) ...")
    fig = generate_province_taylor(acc, station_locs, output_dir)
    if fig is not None:
        plt.close(fig)

    # Per-dekad diagrams (summary + individual)
    print("\nStep 7/8: Per-dekad Taylor diagrams ...")
    generate_per_dekad_taylor_diagrams(
        acc_by_dekad, station_locs, output_dir,
        individual=True, summary=True,
    )

    # Station-level (pooled only; per-dekad already done in step 7)
    print("\nStep 8/8: Station-level Taylor diagram (pooled) ...")
    fig = generate_station_taylor(acc, station_locs, output_dir)
    if fig is not None:
        plt.close(fig)

    print(f"\nAll Taylor diagrams saved to: {output_dir}")
    return acc, station_locs, acc_by_dekad
