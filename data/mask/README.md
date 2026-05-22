# `data/mask/`

Land-sea masks and AOI boundary subsets used by the framework to zero out ocean cells and to clip plotting extents.

## Subfolders

```
data/mask/
├── aoi/        # User AOI masks (committed when small; Bali ships here)
├── iso3/       # Country masks via ISO-3 code (Indonesia tracked; others local)
└── world/      # Global land/sea masks (not committed - too large)
```

### `aoi/` - tracked

User-defined AOI subsets, written by `notebooks/00_define_aoi.ipynb` (bounding box, country, or user polygon flavours). The Bali subset (`bali_subset.nc`) and a generic `aoi_land_mask.nc` ship in the repo. Add your own AOI mask here when adapting to a new region.

### `iso3/` - partially tracked

Country-level land masks generated from BPS / Natural Earth boundaries. Indonesia (`idn_subset.nc`, `iso3_land_mask.nc`) is tracked; other countries are gitignored to keep repo size down.

### `world/` - placeholder

Global land-sea masks at IMERG resolution (~700 MB). Not committed - download from your preferred land-mask source if you need it. For the framework's purposes, the `aoi/` mask is enough.

## File format

CF-1.8 NetCDF with a `land` variable: `1` = land, `0` or `NaN` = sea. Same `lat`/`lon` grid as the precipitation files (0.1° for IMERG, 0.5° for CPC-native).

## Pointed at by config

`config.yml` → `mask_file: '{input_dir}/mask/aoi/{prefix}_mask.nc'` (resolved at runtime via `initialize_config`).
