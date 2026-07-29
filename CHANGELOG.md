# Changelog

Release notes for the `hybrid-bias-correction` framework. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows
[DateVer](https://github.com/datever/datever): `vYYYY.MM` for releases,
`vYYYY.MM.N` for patches within a release.

The site-rendered version of this file is at
[docs/changelog.qmd](docs/changelog.qmd).

---

## v2026.07 - 2026-07-29 - correctness and provenance

Corrections found while reconciling the documentation site against the code
and against the thesis value ledger. No change to the correction algorithm
itself, so results from v2026.05 remain valid.

### Fixed

- **Batch loops ran a fraction of the year.** The batch cells in
  `notebooks/02_lseqmdl_bias_correction.ipynb` and
  `notebooks/03_measuring_performances.ipynb` were left at development
  ranges, covering 9 and 3 of the 36 dekadal windows while reporting that
  they covered all 36. Both now run the full year.
- **`existing_model_action` was ignored.** The setting was read from
  `config.yml` but never consulted, so `overwrite` silently reused a cached
  model. It is now honoured, including `abort`.
- **Config changes made after import were ignored.** `apply_deeplearning_model`
  froze `blend_alpha` as a default argument at import time, so a value set by
  a later `initialize_config()` had no effect unless the caller patched the
  function object. Both `blend_alpha` and the GPD threshold percentile are now
  resolved when the function runs.
- **Stale `saturation_count` documentation.** Docstrings and a `config.yml`
  comment still described a default of 3; the shipped default is 2.

### Added

- **CF and provenance metadata on corrected outputs.** Corrected precipitation
  files now carry `Conventions: CF-1.8` and a `references` DOI, matching the
  metrics, quality and mask writers, plus `run_timestamp`, `framework_version`,
  `git_commit`, `blend_alpha`, `gpd_threshold_percentile` and
  `saturation_count`. Files in the archived Indonesia bundle predate this and
  do not carry them.
- The precipitation variable no longer carries a `standard_name`. The previous
  value, `corrected_precipitation`, is not in the CF standard-name table, and CF
  treats the attribute as optional: omitting it is correct when no controlled
  term applies. `long_name` carries the description and the correction method is
  in the filename and title. `units` now read `mm/day`, following the IMERG-L
  source. Data values are unchanged.

### Changed

- **Zenodo citations now use all-versions DOIs**, so they follow each release
  rather than pinning to the first one: `10.5281/zenodo.20473507` for the
  software and `10.5281/zenodo.20287846` for the data.
- **Documentation corrected throughout.** Function names that no longer
  existed, output filename patterns, the CNN architecture listing (which
  omitted two pooling layers), the metric inventory (which claimed KGE,
  Spearman and mean error, none of which are computed), runtime figures, the
  validated-station count, and correlation values that had drifted from the
  value ledger.

### Companion publications

- Journal manuscript on the methodology and Indonesia validation: **published**
  in *Remote Sensing* 2026, 18, 2298
  ([doi:10.3390/rs18142298](https://doi.org/10.3390/rs18142298)). This
  supersedes the under-review status recorded under v2026.05.

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
  repo and runs in about 72 minutes across notebooks 02 to 06 on a free
  Colab CPU.
- **Indonesia data bundle** - 2001-2025 inputs, masks, and outputs
  (~42 GB) published at
  [Zenodo 10.5281/zenodo.20287846](https://doi.org/10.5281/zenodo.20287846).
- **Documentation site** - Methodology, Implementation, Tutorials, Bali
  Results, FAQ, and auto-generated API reference, rendered with Quarto.
- **Tests** (`tests/`) - smoke test exercising each pipeline stage on tiny
  synthetic arrays; runnable on free CI without the data archive.

### Companion publications

- Journal manuscript on the methodology and Indonesia validation: under
  review at *Remote Sensing* (MDPI).
  
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
