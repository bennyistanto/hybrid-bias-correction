# `data/example_bali/output/`

Placeholder for the **Bali example outputs**. This directory is gitignored - running [`notebooks/example_bali/02_lseqmdl_bias_correction_bali.ipynb`](../../../notebooks/example_bali/) onwards populates it.

A complete run produces ~36 dekads × 3 methods + metrics + QA + station validation + figures ≈ several hundred MB. All regenerable from the inputs in [`../`](../) within ~15 minutes on a free Colab CPU.

## Expected layout

```
output/
├── corrected_ls/, corrected_lseqm/, corrected_lseqmdl/
│   └── bali_cli_{stage}_corrected_imergl_month{MM}_dekad{DD}.nc4  (36 each)
├── trained_models/
│   └── bali_cli_cnn_month{MM}_dekad{DD}.keras                     (36 files)
├── metrics_{ls,lseqm,lseqmdl}/
│   └── bali_cli_metricssd_cpc_imergl_{stage}_month{MM}_dekad{DD}.nc4
├── quality_{ls,lseqm,lseqmdl}/
│   └── bali_cli_qualitysd_cpc_imergl_{stage}_month{MM}_dekad{DD}.nc4
├── station_density/
│   └── bali_cli_confidence_mask_station_density.nc4
├── station_validation/
│   └── bali_cli_station_validation_{stage}.csv                    (3 files)
└── figures/                                                       (nb04-06 PNGs)
```

## How to pre-view results without running

A subset of these figures is embedded in the docs site (see [`../README.md`](../README.md) for links). The executed notebooks under [`docs/example-bali/`](../../../docs/example-bali/) show full pipeline output inline.
