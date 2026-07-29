# `data/output/`

Everything the bias-correction pipeline (nb02-nb06) writes. **Placeholder in the repo - actual files are generated locally.** When you run the pipeline, this directory fills with NetCDFs, CSVs, and PNGs per the layout below.

For Indonesia, the published results from the paper are on Zenodo: <https://doi.org/10.5281/zenodo.20287846> (~40 GB). Extract them here to inspect the paper's exact outputs without re-running anything.

## Expected layout (after running nb02-nb06)

```
data/output/
├── corrected_ls/                        # LS-corrected precipitation NetCDFs
├── corrected_lseqm/                     # LSEQM-corrected precipitation NetCDFs
├── corrected_lseqmdl/                   # LSEQM+DL corrected precipitation NetCDFs
├── trained_models/                      # Per-dekad CNN models (.keras)
├── metrics_ls/                          # Per-method WMO metrics NetCDF
├── metrics_lseqm/
├── metrics_lseqmdl/
├── quality_ls/                          # Per-method CQI NetCDF (basic / dist / temporal)
├── quality_lseqm/
├── quality_lseqmdl/
├── station_density/                     # Confidence mask NetCDF (gauge-density gating)
├── station_validation/                  # Per-station validation CSVs
└── figures/                             # PNG / PDF outputs from nb04, nb05, nb06
    ├── qa/                              # CQI maps, component box plots
    ├── station_validation/              # Per-station metric maps, threshold curves
    ├── taylor/                          # Taylor diagrams (domain / island / province / station)
    └── paper/                           # Compilation figures referenced by the manuscript
```

## File naming convention

`{prefix}_{stage}_corrected_imergl_month{MM}_dekad{DD}.nc4`

- `{prefix}` = `filename_prefix` from your config (e.g. `idn_cli`)
- `{stage}` ∈ `ls`, `lseqm`, `lseqmdl`
- `{MM}` ∈ `01`..`12`
- `{DD}` ∈ `01` (days 1-10), `11` (days 11-20), `21` (days 21-end)

36 dekads per year × 3 methods = 108 NetCDFs per `corrected_*/` for a full run.

## Bali equivalent

When `CONFIG_FILE = 'config_bali.yml'`, outputs land under [`../example_bali/output/`](../example_bali/output/) instead (also gitignored - regenerable by re-running the notebooks).

## Why not commit?

Indonesia output bundle is ~40 GB; even the Bali subset is hundreds of MB. Outputs are regenerable from inputs in well under a day on Colab, so storing them in Git would be pure overhead.
