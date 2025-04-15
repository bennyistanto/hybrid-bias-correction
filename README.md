# Hybrid Bias Correction

Hybrid Bias Correction (LSEQM+DL) is a Python-based workflow for correcting biases in daily precipitation data. It combines traditional methods—Linear Scaling (LS) and Empirical Quantile Mapping (EQM)—with an optional Deep Learning (DL) component to improve the correction, particularly for extreme events.

![LSEQM+DL](docs/images/lseqmdl.png "LSEQM+DL")

## Documentation

Visit the documentation via this link: [https://bennyistanto.github.io/hybrid-bias-correction](https://bennyistanto.github.io/hybrid-bias-correction)

:::{admonition} Note!
:class: note

*This is a living document that will be updated as we develop and refine our methodology for daily precipitation bias correction. Feedback and contributions are welcome.*

:::

## Prerequisites

Before using this project, you should be comfortable with:

- **Python and Jupyter Notebooks:** Basic programming skills and familiarity with running/notebook cells.
- **Basic Statistics and Climate Analytics:** An understanding of fundamental statistical concepts (e.g., means, distributions, quantiles) and an awareness of issues in climate data analysis.
- **Gridded Data Processing:** Knowledge of working with multidimensional datasets (e.g., NetCDF files) for spatial and temporal analysis.

This project is designed to work with gridded daily precipitation timeseries in NetCDF format for both satellite and observational data.

## Publication

to be add

## Contributing

Contributions and suggestions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the Mozilla Public License 2.0.
See LICENSE for the full license text or visit [https://www.mozilla.org/en-US/MPL/2.0/](https://www.mozilla.org/en-US/MPL/2.0/).

## Authors

**Benny Istanto** - [https://benny.istan.to/about](https://benny.istan.to/about)

- Geospatial Operations Support Team, DEC Data Group, The World Bank, United States ([bistanto@worldbank.org](mailto:bistanto@worldbank.org))
- Applied Climatology Study Program, Bogor Agricultural University, Indonesia ([bennyistanto@ipb.ac.id](mailto:bennyistanto@ipb.ac.id))

With supervision from [Prof. Rizaldi Boer](https://scholar.google.co.id/citations?user=jTPXEp8AAAAJ&hl=id) and [Dr. I Putu Santikayasa](https://scholar.google.com/citations?user=DcQ58z8AAAAJ&hl=en) as part of the MSc theses.
