# Hybrid Bias Correction

**Values, Distributions, Extremes - with Neural Refinement.**

*Adjusting values, aligning distributions, preserving extremes - with neural refinement where station density allows.*

A reproducible Python framework for daily satellite precipitation bias correction. Combines **Linear Scaling** (mean adjustment), **Empirical Quantile Mapping with a Generalized Pareto tail** (distribution and extreme alignment), and a lightweight **CNN refinement** (spatial polish gated by station-density confidence). Operationally tested over Indonesia (2001-2025) on the 0.1 deg IMERG grid.

![LSEQM+DL](docs/images/lseqmdl-banner.png "LSEQM+DL")

[![DOI (code)](https://zenodo.org/badge/DOI/10.5281/zenodo.20473508.svg)](https://doi.org/10.5281/zenodo.20473508)
[![DOI (data)](https://img.shields.io/badge/data-10.5281%2Fzenodo.20287847-blue.svg)](https://doi.org/10.5281/zenodo.20287847)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-FF7139.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

## Documentation

Full documentation site: [https://bennyistanto.github.io/hybrid-bias-correction](https://bennyistanto.github.io/hybrid-bias-correction)

| Section | What's there |
|---------|--------------|
| [Methodology](https://bennyistanto.github.io/hybrid-bias-correction/methodology/) | Theory behind each correction stage |
| [Implementation](https://bennyistanto.github.io/hybrid-bias-correction/implementation/) | Algorithm view: which `src/` function does what |
| [Tutorials](https://bennyistanto.github.io/hybrid-bias-correction/tutorials/) | Full pipeline walkthroughs against the Bali example |
| [Executed Notebooks](https://bennyistanto.github.io/hybrid-bias-correction/example-bali/) | nb00-nb06 rendered with all outputs intact (data prep + Bali pipeline) |
| [FAQ](https://bennyistanto.github.io/hybrid-bias-correction/faq.html) | Basics + honest answers to reviewer-style questions |
| [API Reference](https://bennyistanto.github.io/hybrid-bias-correction/technical/api-reference/) | Auto-generated module docs |

## What ships in this repo

- `src/` - The framework as a Python package.
- `notebooks/` - Five-step end-to-end pipeline (`02_lseqmdl_bias_correction` through `06_visualisation_hub`) plus the optional data-acquisition and paper-results notebooks.
- `data/example_bali/` - 11 MB Bali example bundle (IMERG-L, IMERG-F, CPC-UNI at 0.1 deg and native 0.5 deg, BMKG stations, land/sea mask). Runs end to end in ~15 minutes on a free Colab CPU.
- `data/mask/aoi/bali_subset.nc` - The Bali AOI definition.
- `docs/` - Quarto site source.
- `config.yml` (Indonesia) and `config_bali.yml` (Bali example) - the two driver configs.

## Full-Indonesia data archive (Zenodo)

The repository ships the small Bali example bundle so the pipeline is runnable out of the box. The full-Indonesia operational input (IMERG + CPC + BMKG, 2001-2025, ~1.7 GB) and outputs (corrected NetCDFs, metrics, QA, station validation, figures, ~40 GB) are too large for the repository and are deposited separately on Zenodo with a citable DOI:

[https://doi.org/10.5281/zenodo.20287847](https://doi.org/10.5281/zenodo.20287847)

To reproduce the Indonesia results: clone this repo, download the Zenodo bundle, point `config.yml` at the extracted directories, and run the notebooks.

The framework source code itself (this repository, at the published release tag) is archived on Zenodo separately at [https://doi.org/10.5281/zenodo.20473508](https://doi.org/10.5281/zenodo.20473508). See "How to cite" below.

## Quickstart

**Google Colab is the recommended way to run this framework.** TensorFlow is pre-installed there, the GPU is provisioned, and the CNN refinement step works out of the box. Skip the install headaches.

Click the Colab badge on any tutorial page in the [docs site](https://bennyistanto.github.io/hybrid-bias-correction/tutorials/), or open the main notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bennyistanto/hybrid-bias-correction/blob/main/notebooks/02_lseqmdl_bias_correction.ipynb)

The notebooks open with `CONFIG_FILE = 'config_bali.yml'` at the top of Step 1. Run cells top to bottom and the Bali example pipeline produces three corrected NetCDFs (LS, LSEQM, LSEQM+DL) for each of 36 dekads.

To run the full Indonesia pipeline: flip `CONFIG_FILE = 'config.yml'` in each of nb02-nb06 and point `config.yml` at the Zenodo data extracted on your machine.

### Local install (if you really need it)

```bash
git clone https://github.com/bennyistanto/hybrid-bias-correction.git
cd hybrid-bias-correction
mamba env create -f environment.yml
mamba activate climate

# Verify TensorFlow before running anything that uses the CNN step:
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"

jupyter lab notebooks/02_lseqmdl_bias_correction.ipynb
```

If the TensorFlow check fails, the LS and LSEQM stages will still run but the LSEQM+DL refinement will not. Either fix the TF install (see [Troubleshooting](https://bennyistanto.github.io/hybrid-bias-correction/user-guide/troubleshooting.html)) or switch to Colab.

## Prerequisites

- Python 3.10 or newer (3.11 recommended).
- Familiarity with Jupyter notebooks and `xarray` / NetCDF.
- Basic understanding of precipitation statistics (quantile mapping, extreme value theory) helps but is not required.

## Validation and testing

The framework's quality is verified along three independent dimensions:

- **Held-out gauge validation** (scientific): the corrected output is
  scored against 171 BMKG stations that are not used in the correction.
  Across the four verification pillars (value adjustment, distribution
  alignment, extreme preservation, event detection), the LSEQM+DL output
  moves three pillars cleanly toward the gauge target and produces a
  designed trade-off in the fourth. See
  [docs/technical/validation.qmd](docs/technical/validation.qmd) and
  Chapter 4 of the thesis for the full station-level results.
- **Sensitivity to exposed parameters** (scientific): across fifteen
  combinations of `blend_alpha`, `gpd_threshold_percentile`, and
  `saturation_count` on the Bali subdomain, the Pearson correlation
  stays in the narrow band [0.332, 0.348] and none of the settings
  reverses the headline pattern. See
  [docs/technical/sensitivity-analysis.qmd](docs/technical/sensitivity-analysis.qmd).
- **Code regression** (engineering): a synthetic-data smoke suite in
  `tests/` exercises the import surface, the distribution-fitting
  primitives, the station-density confidence machinery, and the
  blending algebra. The suite runs in under a second on a free CI
  runner and gates every push via
  `.github/workflows/test.yml`. Local run:
  `python -m pytest tests/ -v`. See
  [docs/technical/testing.qmd](docs/technical/testing.qmd).

## Publication

Companion manuscript: under review at *Remote Sensing* (MDPI). DOI and citation will be added on acceptance.

## How to cite

If you use this framework, please cite the archived software release. The recommended citation is:

> Istanto, B., Boer, R., & Santikayasa, I. P. (2026). *Hybrid Bias Correction: Values, Distributions, Extremes - with Neural Refinement* (v2026.05). Zenodo. [https://doi.org/10.5281/zenodo.20473508](https://doi.org/10.5281/zenodo.20473508)

If you use the bundled full-Indonesia data archive, please cite it separately:

> Istanto, B., Boer, R., & Santikayasa, I. P. (2026). *Hybrid Bias Correction of IMERG Late Run V07 over Indonesia (2001-2025): Input Data, Land-Sea Masks, and Corrected Products* (Version 1) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.20287847](https://doi.org/10.5281/zenodo.20287847)

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders this as a "Cite this repository" button on the repo landing page.

## Contributing

Contributions and suggestions are welcome. Please open an issue or pull request. 

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE) or [https://www.mozilla.org/en-US/MPL/2.0/](https://www.mozilla.org/en-US/MPL/2.0/).

## Authors

**Benny Istanto** - [https://benny.istan.to/about](https://benny.istan.to/about)

  Applied Climatology Study Program, Department of Geophysics and Meteorology</br>
  [Bogor Agricultural University](https://www.ipb.ac.id/), Indonesia</br>
  bennyistanto@apps.ipb.ac.id

With supervision from [Prof. Rizaldi Boer](https://scholar.google.com/citations?user=jTPXEp8AAAAJ&hl=en) and [Dr. I Putu Santikayasa](https://scholar.google.com/citations?user=DcQ58z8AAAAJ&hl=en) as part of the [MSc thesis](https://github.com/bennyistanto/msc-thesis).
