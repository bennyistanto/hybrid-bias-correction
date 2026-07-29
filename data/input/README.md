# `data/input/`

Stitched multi-year NetCDF inputs for the **full Indonesia** pipeline (or your own AOI). This is what `config.yml` points at when `CONFIG_FILE = 'config.yml'` is selected in the notebooks.

**This folder is a placeholder in the repo - the actual files are not committed.** They are too large (~1.7 GB total for Indonesia 2001-2025). Two ways to populate it:

## Option A - Indonesia from Zenodo

The Indonesia operational inputs are deposited on Zenodo: <https://doi.org/10.5281/zenodo.20287846>.

Download the `input_*.tar` archives, extract them, and place the contents so they match the layout below. `config.yml` paths assume this exact structure.

## Option B - Your own AOI from nb01

Run `notebooks/01_data_acquisition.ipynb` against your AOI. It downloads from NASA Earthdata + NOAA PSL and writes the stitched products here.

## Expected layout

```
data/input/
├── imergl/{prefix}_imergl.nc4              # IMERG Late Run V07, daily, 0.1°
├── imergf/{prefix}_imergf.nc4              # IMERG Final Run V07, daily, 0.1°
├── cpcuni/{prefix}_cpcuni.nc4              # CPC-UNI regridded to 0.1°
├── cpcuni/{prefix}_cpcuni_native05.nc4     # CPC-UNI native 0.5° (BCSD-style fitting)
└── stations/{prefix}_stations_location.csv  # WMO ID + lat/lon/elevation
    {prefix}_stations_data.csv              # Daily obs, wide format (column per WMO ID)
```

`{prefix}` matches `general.filename_prefix` in your config (e.g. `idn_cli` for Indonesia, `bali_cli` for the Bali example).

## Bali equivalents

The Bali example bundle (under [`../example_bali/`](../example_bali/)) is the same five files, pre-stitched and clipped to Bali. You don't need anything in this directory to run the Bali tutorial - `config_bali.yml` reads from `example_bali/` directly.

## CF conventions

All NetCDFs are CF-1.8: `lat`/`lon` (descending lat allowed), `time` as `days since 1970-01-01`, precipitation in mm/day. The framework will flip lat to ascending on read if needed (see `src.utility`).
