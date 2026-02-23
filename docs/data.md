# Data

Developing a reliable bias correction framework demands the careful selection of datasets that represent both satellite-based estimates and ground-based observations. This section describes the data sources utilized and outlines the preprocessing steps undertaken to harmonize their spatial and temporal characteristics.

This research only use publicly available data on the internet and utilize free and open source software, mainly [Python](https://www.python.org/) and the libraries.

There are two data will be used in this analysis.

1. **IMERG:**
   The Global Precipitation Measurement (GPM) Integrated Multi-satellite Retrievals for GPM ([IMERG](https://gpm.nasa.gov/data/imerg)) dataset is a high-resolution satellite product that estimates global precipitation. IMERG combines data from multiple satellites to offer near real-time precipitation measurements, with a spatial resolution of 0.1-degrees and a temporal resolution of 30-minutes, daily and monthly.

   This dataset is particularly valuable for monitoring and analyzing precipitation patterns, especially in regions with sparse ground-based observations. IMERG data are accessible via the NASA Goddard Earth Sciences Data and Information Services Center ([GES DISC](https://disc.gsfc.nasa.gov/#!)) and available in GEE Data Catalog for large-scale analyses.

   1. **IMERG Late Run** (IMERG-L) will be used for the analysis.

        IMERG Late Run offers an optimal balance between latency and data quality, providing near-real-time precipitation estimates with a delay of approximately 14 hours. Unlike the IMERG Final Run, which incorporates station data into its post-processing, the Late Run product relies purely on satellite observations, making it ideal for evaluating bias correction methods without the influence of gauge data assimilation. IMERG data are available at a high spatial resolution of 0.1 degrees (~10 km) and are directly accessed from the official NASA Goddard Earth Sciences Data and Information Services Center (GES DISC).

   2. **IMERG Final Run** (IMERG-F) will be used for complementary evaluation.

       In addition to the IMERG Late Run data used for bias correction, this study also utilizes the IMERG Final Run product as a complementary evaluation dataset. IMERG Final Run incorporates gauge-based observations during its post-processing, aiming to further improve precipitation accuracy relative to purely satellite-derived estimates. Although it benefits from station assimilation, the Final Run is not used in the bias correction procedure itself to maintain independence between correction inputs and validation references. Instead, the IMERG Final Run is employed for comparative analysis, providing an additional benchmark to assess the effectiveness of the bias correction applied to the IMERG Late Run data. IMERG Final Run data are also available at a 0.1-degree spatial resolution and were obtained directly from the NASA GES DISC.

   ![IMERG](./images/imerg-20250412.png)

2. **CPC-UNI:**
   The Climate Prediction Center Unified Gauge-Based Analysis of Daily Precipitation (CPC-UNI) dataset is a reliable source of gauge-based precipitation measurements, providing daily precipitation data with a global coverage at a 0.5-degree spatial resolution. This dataset is derived from thousands of rain gauges worldwide and is used to validate and correct satellite precipitation estimates.

   The CPC-UNI dataset is available from the [NOAA Physical Science Laboratory](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html) and serves as an essential reference for bias correction, ensuring the accuracy and reliability of precipitation data in hydrological and climate studies.

   ![CPCUNI](./images/cpcuni-20250410.png)

3. **Independent Validation Data (BMKG)**

    Independent validation is conducted using ground-based observations from Indonesia's Meteorological, Climatological, and Geophysical Agency (BMKG). BMKG operates an extensive network of weather stations providing daily precipitation measurements across the Indonesian archipelago. These independent station data are not used in the bias correction process itself but serve to evaluate the performance of the corrected IMERG precipitation estimates. Validation focuses on key criteria, including the correct detection of rainfall events, the ability to represent observed extremes, and the assessment of spatial consistency between corrected satellite estimates and ground truth. The use of independent validation data ensures an objective assessment of correction effectiveness, particularly for extreme rainfall conditions critical to disaster risk management. BMKG data were accessed directly from the agency’s publicly available online repository.

## Preprocessing and Harmonization

To ensure comparability between the satellite-derived and ground-based datasets, several preprocessing steps were undertaken. First, the CPC-UNI dataset, originally available at a coarser 0.5-degree spatial resolution, was regridded to match the 0.1-degree resolution of the IMERG dataset using nearest-neighbor interpolation. Temporal alignment was performed by synchronizing daily time steps across the datasets and ensuring a strictly monotonic, duplicate-free time axis. A land-sea mask, based on official land boundary datasets, was applied to both IMERG and CPC-UNI to retain only land-based precipitation estimates. These preprocessing steps ensure that both datasets are spatially and temporally consistent, enabling robust application of bias correction techniques and minimizing errors due to resolution or coverage discrepancies.
