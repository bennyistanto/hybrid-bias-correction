# Changelog

Release notes for the `hybrid-bias-correction` framework. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows
[DateVer](https://github.com/datever/datever): `vYYYY.MM` for releases,
`vYYYY.MM.N` for patches within a release.

The site-rendered version of this file is at
[docs/changelog.qmd](docs/changelog.qmd).

---

## v2026.05 - 2026-05-31 - first public release

First public release of the Hybrid Bias Correction framework. This is the
codebase developed for the MSc thesis on daily satellite precipitation bias
correction over Indonesia, refactored for the broader community.

### Added

- **The framework** (`src/`) - Linear Scaling, Empirical Quantile Mapping
  with a Generalized Pareto tail, and a CNN refinement gated by
  station-density confidence. Implemented with one driver config per region.
- **Notebooks** (`notebooks/`) - end-to-end pipeline from AOI definition
  and data acquisition through bias correction, metrics, QA, station
  validation, and visualisation.
- **Bali example bundle** (`data/example_bali/`, 11 MB) - ships with the
  repo and runs end-to-end in ~15 minutes on a free Colab CPU.
- **Indonesia data bundle** - 2001-2025 inputs, masks, and outputs
  (~42 GB) published at
  [Zenodo 10.5281/zenodo.20287847](https://zenodo.org/records/20287847).
- **Documentation site** - Methodology, Implementation, Tutorials, Bali
  Results, FAQ, and auto-generated API reference, rendered with Quarto.
- **Tests** (`tests/`) - smoke test exercising each pipeline stage on tiny
  synthetic arrays; runnable on free CI without the data archive.

### Companion publications

- MSc thesis: *How Far Can Bias Correction Improve Daily Satellite
  Precipitation?* (Istanto, IPB University, 2026).
- Journal manuscript on the methodology and Indonesia validation: under
  review at *Remote Sensing* (MDPI).
- Journal of Open Source Software paper on the framework itself
  (in preparation; draft at `paper/joss/`).

---

## Backlog

- Replace the CNN architecture with a U-Net to avoid the Flatten -> Dense
  bottleneck.
- Formal sensitivity analysis over `blend_alpha`,
  `gpd_threshold_percentile`, and `saturation_count`.
- Extend continuous-integration coverage from the smoke test to a Bali
  end-to-end run on each release.
- `tf.keras.backend.clear_session()` between dekads in nb02 for memory
  predictability on Indonesia-scale runs.
