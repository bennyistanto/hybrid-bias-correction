# `data/downloads/`

Working area for `notebooks/01_data_acquisition.ipynb`. Raw per-day files are written here when you adapt the framework to a new region.

**This folder is a placeholder in the repo - the actual files are not committed.** They are generated locally by nb01 (and stay local, since they're large and can always be re-downloaded).

## Expected layout (after running nb01)

```
data/downloads/
├── GPM_3IMERGDL/          # IMERG Late Run V07 HDF5 per-day files (raw)
├── GPM_3IMERGDL_extract/  # Decoded NetCDFs (one per day)
├── GPM_3IMERGDL_subset/   # Clipped to AOI (one per day)
├── GPM_3IMERGDF/          # IMERG Final Run V07, same structure
├── GPM_3IMERGDF_extract/
├── GPM_3IMERGDF_subset/
├── CPC_UNI/               # CPC-UNI annual NetCDFs (global, 0.5°)
├── CPC_UNI_extract/       # Per-day NetCDFs cropped to your AOI window
├── boundary/              # Natural Earth / geoBoundaries / BPS shapefiles
└── userdata/              # User-provided shapefile if nb00 used a custom polygon
```

nb01 streams from these into the **stitched** multi-year NetCDFs in [`../input/`](../input/) - that's what the bias-correction pipeline (nb02+) actually reads.

## When can I delete the contents?

After nb01 finishes and `data/input/` contains the stitched products, the per-day files in `downloads/` are no longer needed. Keep them only if you intend to re-clip to a different AOI without re-downloading from NASA / NOAA.

## Why not commit them?

Per-day IMERG covers 2001-now at ~5 MB/day. That's tens of GB before clipping - too large for a Git repo and freely re-downloadable from the source.
