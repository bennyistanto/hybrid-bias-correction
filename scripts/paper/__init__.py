"""
scripts.paper — Paper-only helpers for PPS1591 LSEQM+DL paper.

This package is NOT part of the reproducible research release. It contains
post-hoc analyses and table builders used to generate figures and numbers
for the paper draft, while keeping `src/` clean.

Modules
-------
- paper_helpers       : common loaders, file paths, METHODS constants
- dekad_aggregation   : dekad-aggregated metrics (B2 — temporal-scale fix)
- timezone_diagnostic : day-shift lag tests (B1) and CPC convention check (B3)
- table_builders      : regenerate Tables 4, 5, 6, 7 with corrected framing
"""

from .paper_helpers import (
    METHODS, METHOD_LABELS, ALL_PERIODS, DEKAD_MAP,
    metrics_path, station_val_path,
    load_corrected_da, load_reference_da, load_station_obs,
    load_station_locations_df,
    PROJECT_ROOT,
)
from .dekad_aggregation import (
    aggregate_to_dekad_totals,
    aggregate_station_df_to_dekad,
    compute_dekad_grid_metrics,
    compute_dekad_station_metrics,
    compute_stratified_dekad_grid_metrics,
    compute_stratified_dekad_station_metrics,
)
from .timezone_diagnostic import (
    day_shift_lag_test_station,
    day_shift_lag_test_grid,
    cpc_imerg_convention_test,
)
from .table_builders import (
    build_table4_grid,
    build_table7_station,
    build_ks_pixel_passing_fraction,
)

__all__ = [
    'METHODS', 'METHOD_LABELS', 'ALL_PERIODS', 'DEKAD_MAP',
    'metrics_path', 'station_val_path',
    'load_corrected_da', 'load_reference_da', 'load_station_obs',
    'PROJECT_ROOT',
    'aggregate_to_dekad_totals',
    'compute_dekad_grid_metrics',
    'compute_dekad_station_metrics',
    'day_shift_lag_test_station',
    'day_shift_lag_test_grid',
    'cpc_imerg_convention_test',
    'build_table4_grid',
    'build_table7_station',
    'build_ks_pixel_passing_fraction',
]
