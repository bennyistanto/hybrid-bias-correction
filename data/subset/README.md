# `data/subset/`

Administrative boundary subsets used only for **plotting overlays** (province lines on QA maps, island grouping for Taylor diagrams, etc.). They do not affect any computation.

## Subfolders

```
data/subset/
├── adm0/    # Country boundaries (placeholder)
├── adm1/    # Province / state boundaries (placeholder)
├── bnd/     # Custom boundary subsets (placeholder)
├── iso3/    # ISO-3 derived boundaries - Indonesia tracked
└── world/   # Global boundary archives (placeholder)
```

### `iso3/` - tracked

Indonesia province polygons (`idn_admin1.nc`, `idn_region.nc`, `idn_subset.nc`) ship in the repo. They drive the per-island Taylor diagrams and the regional box plots in nb05/nb06.

### Other subfolders - placeholder

Adapting to a different country? Drop boundary files (NetCDF, shapefile, or GeoJSON) into `adm0/` or `adm1/` and point your config at them via the `visualisation.boundaries.*` keys. Default cartopy lines are used if no override is provided - see `src.visualisation._add_boundaries`.

## Why not committed in full

`data/downloads/boundary/` (gitignored) holds the raw shapefiles. Only the clipped, framework-ready NetCDFs land here, and even those grow quickly with high-res world coverage. The Indonesia files are the only ones needed for the published paper.
