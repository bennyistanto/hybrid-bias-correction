# `data/example_bali/`

The **Bali example bundle** that ships with the repo (~11 MB). It is everything the bias-correction pipeline needs to run end-to-end against a small AOI in under 15 minutes on a free Colab CPU.

`config_bali.yml` at the repo root points at this directory. The notebooks default to `CONFIG_FILE = 'config_bali.yml'` so you can run the full pipeline without downloading anything.

## What's in here

| File | Variable | Period | Grid |
|------|----------|--------|------|
| `bali_imergl.nc4` | `precipitation` (mm/day) | 2001-2025 | 0.1° (IMERG Late Run V07) |
| `bali_imergf.nc4` | `precipitation` (mm/day) | 2001-2025 | 0.1° (IMERG Final Run V07) |
| `bali_cpcuni.nc4` | `precip` (mm/day) | 2001-2025 | 0.1° (CPC-UNI regridded) |
| `bali_cpcuni_native05.nc4` | `precip` (mm/day) | 2001-2025 | 0.5° (CPC-UNI native) |
| `bali_mask.nc` | `land` (0/1) | n/a | 0.1° (Bali AOI bounding box) |
| `bali_stations_location.csv` | WMO ID + lat/lon/elev | n/a | 4 BMKG stations |
| `bali_stations_data.csv` | daily mm/day per station | 2001-2025 | 4 BMKG stations |

The 4 BMKG stations: WMO 97230 (I Gusti Ngurah Rai), 97232 (Geofisika Denpasar), 97234 (Kahang-Kahang), 97236 (Klimatologi Jembrana).

## Outputs - `output/`

The `output/` subfolder is **not committed** (`.gitignore`'d as `data/example_bali/output/`). Running the Bali pipeline writes:

```
data/example_bali/output/
├── corrected_ls/, corrected_lseqm/, corrected_lseqmdl/   # 36 NetCDFs each
├── trained_models/                                        # 36 .keras files
├── metrics_{ls,lseqm,lseqmdl}/                            # 36 NetCDFs each
├── quality_{ls,lseqm,lseqmdl}/                            # 36 NetCDFs each
├── station_density/                                       # Confidence mask
├── station_validation/                                    # Per-station CSVs
└── figures/                                               # PNGs from nb04/05/06
```

To regenerate: open `notebooks/02_lseqmdl_bias_correction.ipynb` and run all cells. nb03-nb06 chain from these outputs.

## How the bundle was built

See `scripts/build_bali_example.py` (gitignored) for the script that clips the full-Indonesia products to the Bali bounding box (114.45-115.75° E, -8.85 to -8.05° N), including the 1-cell buffer on the CPC native file to avoid convex-hull NaN issues on small AOIs.

## Where the example results live in the docs

Pre-rendered Bali outputs (figures + executed notebooks) are visible on the docs site:

- [docs/example-bali/](../../docs/example-bali/) - the 7 executed notebooks (nb00-nb06) rendered with all outputs intact.
- [docs/images/bali/](../../docs/images/bali/) - 20 representative PNGs (Taylor diagrams, CQI maps, per-station scatter / time series for all 4 stations, threshold curves, regional box plots) embedded into the [QA Framework](https://bennyistanto.github.io/hybrid-bias-correction/tutorials/qa-framework.html) and [Station Validation](https://bennyistanto.github.io/hybrid-bias-correction/tutorials/station-validation.html) tutorial pages.

Inspect either to see what the framework produces before running anything yourself.
