# Hybrid Bias Correction

**Enhancing Daily Satellite Precipitation Estimates:** Adjusting Values, Aligning Distributions, and Preserving Extremes

---

Welcome to the documentation for the **Hybrid Bias Correction** project. This project aims to enhance the accuracy and reliability of satellite-based precipitation estimates by correcting systematic biases and aligning data distributions with ground observations.

The workflow integrates several key components:

- **Linear Scaling (LS):**  
  Corrects the mean bias between satellite estimates and observed data.
  
- **Empirical Quantile Mapping (EQM):**  
  Aligns the entire distribution of the satellite data with that of the observations using a gamma distribution-based approach.
  
- **Tail Adjustment with Generalized Pareto Distribution (GPD):**  
  Enhances the representation of extreme precipitation events.
  
- **Deep Learning (DL) Enhancement:**  
  Refines the bias correction output by targeting extreme values and ensuring spatial consistency that traditional statistical methods might miss.

This documentation is structured to guide you through the methodology, code structure, installation steps, and practical usage of the bias correction workflow. Whether you are a researcher, data scientist, or student, you'll find detailed explanations and usage examples, to help you effectively implement and understand the bias correction process.

## Table of Contents

- [Repository Structure](repository.md)
- [Data](data.md)
- [Methodology](methodology.md)
- [Code Documentation](code_documentation.md)
- [Setup](setup.md)
- [Implementation](implementation.md)

---

## Prerequisites

To effectively use this documentation and run the bias correction workflow, users should be familiar with:

- **Python and Jupyter Notebook:**  
  Basic programming in Python and navigating Jupyter notebooks are essential for running the provided code and modifying examples.
  
- **Basic Statistics:**  
  An understanding of fundamental statistical concepts such as mean, variance, distribution functions, and quantile mapping will help you grasp the bias correction methods.
  
- **Climate Analytics & Earth Observation:**  
  Familiarity with climate datasets, especially gridded, multidimensional data (e.g., NetCDF), is beneficial. The workflow is designed to work with daily precipitation timeseries from satellite (e.g., IMERG) and observational data (e.g., CPC-UNI).
  
- **Data Preprocessing:**  
  Knowledge of geospatial data processing (e.g., regridding, aligning time series) is useful, since consistent data structure and formatting are crucial for bias correction.

---

## Notes & Limitations

- **Data Format:**  
  The hybrid bias correction workflow is designed to operate on gridded precipitation daily timeseries in single file with NetCDF format. This format supports multidimensional computations across both space and time, ensuring that essential metadata (e.g., units, coordinate information, and temporal resolution) is preserved for proper analysis.

- **Data Structure:**  
  Consistency in data structure is critical. The workflow expects that both satellite estimates and ground observations are provided as gridded datasets with matching spatial and temporal dimensions. Preprocessing steps—such as regridding, aligning time indices, and ensuring a monotonic time series—are essential to ensure that the input data conforms to the expected structure for the bias correction process.

- **Test Case:**  
  The current implementation has been tested using [CPC-UNI](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html) as the gridded reference dataset and [IMERG](https://gpm.nasa.gov/data/imerg) as the satellite input. Users employing different datasets or resolutions may need to adjust parameters or perform additional preprocessing steps.

- **Deep Learning Component:**  
  While the DL enhancement refines the corrections for extreme events and spatial variability, its performance is dependent on the quality and representativeness of the training data. Users may need to retrain the model for significantly different data sources or regions.

## Contributing

Contributions and suggestions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the Mozilla Public License 2.0.
See LICENSE for the full license text or visit [https://www.mozilla.org/en-US/MPL/2.0/](https://www.mozilla.org/en-US/MPL/2.0/).

## Authors

**Benny Istanto** - [https://benny.istan.to/about](https://benny.istan.to/about)

- Geospatial Operations Support Team, DEC Data Group, The World Bank, United States ([bistanto@worldbank.org](mailto:bistanto@worldbank.org))
- Applied Climatology Study Program, Graduate School, Bogor Agricultural University, Indonesia ([bennyistanto@ipb.ac.id](mailto:bennyistanto@ipb.ac.id))

With supervision from [Prof. Rizaldi Boer](https://scholar.google.co.id/citations?user=jTPXEp8AAAAJ&hl=id) and [Dr. I Putu Santikayasa](https://scholar.google.com/citations?user=DcQ58z8AAAAJ&hl=en) as part of the MSc theses.

---

:::{admonition} Note!
:class: note

*This is a living document that will be updated as we develop and refine our methodology for daily precipitation bias correction. Feedback and contributions are welcome.*

:::

![LSEQM+DL](./images/lseqmdl.png "LSEQM+DL")
