# Code Documentation

This chapter provides detailed documentation for the source code files located in the `src` folder. Each Python script is organized with a header section explaining its purpose, along with descriptions of the major functions it contains and how they contribute to the overall bias correction workflow.

## __init__.py

The `__init__.py` file marks the `src` folder as a Python package. It contains no functional code but serves as an initialization script that defines the package documentation.

__Key Points:__

- __Purpose:__ Marks the folder as a Python package.
- __Content:__ Contains a brief module docstring describing the Hybrid Bias Correction (LSEQM+DL) project.

## config.py

The `config.py` module centralizes all configurable parameters and file paths for the bias correction workflow. This file defines default directories, input file paths, output directories, a standardized NetCDF filename template, and key parameters for statistical fitting and deep learning training.

__Key Functions/Content:__

- __Directory Settings:__ Defines the main project directory, input, and output directories.
- __File Paths:__ Provides default paths for IMERG, CPC, and land-sea mask files.
- __Output Settings:__ Specifies output directories for LS, LSEQM, DL-corrected products, and trained models.
- __Filename Template:__ A template for generating standardized NetCDF filenames.
- __Statistical Parameters:__ Settings for GPD fitting (number of splits, threshold percentiles, etc.).
- __Deep Learning Parameters:__ Training parameters including epochs, batch size, dropout rates, filter sizes, number of filters, dense layer size, and optimizer.

## bias_correction.py

This module contains the high-level workflow function for bias correction. It orchestrates the process by combining the following steps:

- __Linear Scaling (LS):__ Adjusts the overall magnitude of the IMERG data using the ratio of CPC to IMERG means.
- __Empirical Quantile Mapping (EQM):__ Aligns the distributions via gamma quantile mapping with tail adjustment.
- __Deep Learning (DL) Enhancement:__ Optionally applies a trained DL model to further refine the corrected precipitation, particularly over extreme values.

__Key Functions:__

- `lseqmdf(...)`  
  Executes the entire bias correction process. It accepts IMERG and CPC datasets, performs aggregation, alignment, applies LS, follows with EQM (including gamma-based quantile mapping and GPD tail adjustment), and conditionally applies a DL model to correct extreme precipitation values.

## deep_learning.py

The `deep_learning.py` module handles all functions related to training and applying the deep learning (DL) model for bias correction. The DL component refines the EQM-corrected precipitation by further adjusting pixel-level extreme values.

__Key Functions:__

- `train_bias_correction_model(...)`  
  Trains a Convolutional Neural Network (CNN) on aggregated multi-year data (IMERG and CPC) to learn the mapping between the two datasets. This model serves as a fine-tuning step to further reduce bias, especially for extremes.
- `apply_deeplearning_model(model, imerg_data)`  
  Applies the trained DL model to the EQM-corrected IMERG data. Only pixels exceeding a specified threshold are replaced with DL predictions, preserving the original values for non-extreme observations.

## distribution_fitting.py

This module is dedicated to statistical distribution fitting and quantile mapping essential for the bias correction process. It ensures that the statistical properties of the satellite and observation datasets are properly aligned.

__Key Functions:__

- `calculate_l_moments(data)`  
  Computes L-moments and L-moment ratios, providing robust measures of the distribution shape.
- `fit_gamma_with_l_moments(data)`  
  Uses L-moments to fit a gamma distribution to the precipitation data.
- `fit_generalized_pareto_distribution(data, threshold)`  
  Fits a Generalized Pareto Distribution (GPD) to the excesses above a given threshold.
- `cross_validate_gpd(data, threshold, n_splits)`  
  Performs K-Fold cross-validation to yield stable GPD parameters.
- `gamma_quantile_mapping(imerg_values, cpc_values)`  
  Performs gamma distribution-based quantile mapping with tail adjustment. This function adjusts the entire distribution of IMERG data to match CPC data, explicitly addressing extreme precipitation values using the GPD.

## io.py

The `io.py` module focuses on file input/output operations for the workflow. It provides functions to save corrected precipitation datasets, handle time aggregation, and manage NetCDF file operations.

__Key Functions:__

- `save_corrected_precip(precip_data, ds, method_abbr, method_full, folder, dekad_str, month_str)`  
  Saves the corrected precipitation data to a NetCDF file following CF metadata conventions.
- `get_max_day_in_month(ds, month)`  
  Determines the maximum day in a specified month by scanning across all years in the dataset, accounting for leap years.
- `aggregate_data_across_years(imerg_ds, cpc_ds, month, dekad_start_day, dekad_end_day)`  
  Aggregates multi-year data for a specified dekad from both the IMERG and CPC datasets.

## utility.py

The `utility.py` module provides helper functions that support data preparation and I/O management throughout the bias correction workflow.

__Key Functions:__

- `set_user_decision()`  
  Prompts the user for a decision when a file already exists (Overwrite, Skip, or Abort).
- `apply_land_sea_mask(data, mask_file)`  
  Applies a land-sea mask to the given dataset, ensuring that only terrestrial areas are processed.
- `ensure_strict_monotonic_time(ds)`  
  Processes the time index of a dataset to enforce strict monotonicity and remove duplicates.
- `reindex_and_align_with_monotonicity(reference_ds, secondary_ds, land_sea_mask)`  
  Reindexes and aligns a secondary dataset to a reference dataset (e.g., aligning CPC data with IMERG data) while ensuring a monotonic time index.

---

This documentation chapter provides an overview of each module, its primary functions, and how these functions contribute to the overall bias correction process. Users interested in understanding the inner workings of the code can refer to the respective module files for detailed explanations and implementation details.
