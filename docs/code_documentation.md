# Code Documentation

This chapter provides detailed documentation for the source code files located in the `src` folder. Each Python script is organized with a header section explaining its purpose, along with descriptions of the major functions it contains and how they contribute to the overall bias correction workflow.

```graphql
hybrid-bias-correction/
├── src/
│   ├── __init__.py              # Marks src as a Python package
│   ├── config.py                # Default configuration (paths, parameters, and filename template)
│   ├── utility.py               # Utility functions for data preparation and I/O management
│   ├── distribution_fitting.py  # Functions for calculating L-moments, gamma and GPD fitting, and quantile mapping
│   ├── io.py                    # Functions for file I/O, saving NetCDF files, and data aggregation
│   ├── deep_learning.py         # Functions for training and applying the deep learning model
└── └── bias_correction.py       # High-level bias correction workflow (LS + EQM [+ DL])
```

## Initialization

The `__init__.py` file marks the `src` folder as a Python package. It contains no functional code but serves as an initialization script that defines the package documentation.

__Key Points:__

- __Purpose:__ Marks the folder as a Python package.
- __Content:__ Contains a brief module docstring describing the Hybrid Bias Correction (LSEQM+DL) project.

:::{admonition} Code
:class: dropdown

```python
# __init__.py
# This file marks the directory as a Python package.

"""Hybrid Bias Correction (LSEQM+DL)"""
```

:::

## Configuration

The `config.py` module centralizes all configurable parameters and file paths for the bias correction workflow. This file defines default directories, input file paths, output directories, a standardized NetCDF filename template, and key parameters for statistical fitting and deep learning training.

__Key Functions/Content:__

- __Directory Settings:__ Defines the main project directory, input, and output directories.

    :::{admonition} Code
    :class: dropdown

    ```python
    # Default directories (these can be overridden)
    main_dir = f'/content/drive/MyDrive/hybrid-bias-correction'
    input_dir = f'{main_dir}/data/bc/input'
    output_dir = f'{main_dir}/data/bc/output'
    ```

    :::

- __File Paths:__ Provides default paths for IMERG, CPC, and land-sea mask files.

    :::{admonition} Code
    :class: dropdown

    ```python
    # Default input file paths
    imerg_file = f'{input_dir}/imergl/idn_imergl.nc4' # Single daily timeseries multi-year file
    cpc_file = f'{input_dir}/cpcuni/idn_cpcuni.nc4' # Single daily timeseries multi-year file
    mask_file = f'{main_dir}/data/subset/iso3/idn_subset.nc' # Mask file
    ```

    :::

- __Output Settings:__ Specifies output directories for LS, LSEQM, DL-corrected products, and trained models.

    :::{admonition} Code
    :class: dropdown

    ```python
    # Output directories for different correction products
    ls_corrected_precip_path = f'{output_dir}/corrected_ls' # Directory for storing corrected precip
    lseqm_corrected_precip_path = f'{output_dir}/corrected_lseqm' # Directory for storing corrected precip
    lseqmdl_corrected_precip_path = f'{output_dir}/corrected_lseqmdl' # Directory for storing corrected precip
    trained_models_path = f'{output_dir}/trained_models' # Directory for storing trained models
    ```

    :::

- __Filename Template:__ A template for generating standardized NetCDF filenames.

    :::{admonition} Code
    :class: dropdown

    ```python
    # Output filename template (using .format() syntax)
    """
    Output filename follows this convention (but it's OK to modify):
    idn: Indonesia
    cli: Climate (thematic)
    method_abbr:
        - ls: Linear Scaling method
        - lseqm: Linear Scaling and Empirical Quantile Mapping
        - lseqmdl: Hybrid Deep Learning-Physical (Linear Scaling and Empirical Quantile Mapping) Approach
    corrected: Corrected Precipitation
    imergl: IMERG Late Run
    month{month_str}: The month number as two-digit (e.g., month01, month12)
    dekad{dekad_str}: The dekad (e.g., dekad01, dekad11, dekad21)
    """
    output_filename_template = "{folder}/idn_cli_{method_abbr}_corrected_imergl_month{month_str}_dekad{dekad_str}.nc4"
    
    # The NetCDF output will follow NetCDF Climate and Forecast (CF) Metadata Convention
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html
    # Encoding for CF 1.8 (adjust based on your precision and range requirements)
    cf18_f32 = {'precipitation': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan}}
    ```

    :::

- __Statistical Parameters:__ Settings for GPD fitting (number of splits, threshold percentiles, etc.).

    :::{admonition} Code
    :class: dropdown

    ```python
    # +++++++++++++++++++++++++++++++++++++++++
    # Configurable Parameters
    # +++++++++++++++++++++++++++++++++++++++++
    
    """
    This parameter controls the number of splits used during the cross-validation process for fitting
    the Generalized Pareto Distribution (GPD). Cross-validation helps to assess the robustness and
    stability of the GPD fitting by splitting the data into training and validation sets multiple times.
    
    Recommendation:
    A value of 5 is typically a good balance. It provides a reasonable level of validation without being
    computationally intensive. If your dataset is large and computation time isn't a concern, you could
    experiment with higher values (e.g., 7 or 10). Conversely, if you're working with smaller datasets,
    3 might be more appropriate.
    """
    N_SPLITS_GPD_CROSSVALIDATE = 5
    
    """
    The GPD_THRESHOLD_PERCENTILE sets the percentile threshold for fitting the Generalized Pareto
    Distribution (GPD) to the data. A lower percentile (e.g., 80%) will include more data points in
    the GPD fitting, possibly smoothing the tail of the distribution, while a higher percentile
    (e.g., 96%) will focus the fitting on the most extreme values.
    
    Best Use:
    * Lower Percentile (e.g., 80%): Suitable for capturing a broader range of extremes, including
    moderately extreme events. This can be useful when there is interest in understanding the distribution
    of both moderate and severe extremes.
    * Higher Percentile (e.g., 96%): Focuses the analysis on the most extreme events, making it better
    suited for cases where only the most significant extremes are of interest, such as in studies focused
    on disaster risk or rare weather events.
    """
    GPD_THRESHOLD_PERCENTILE = 80
    
    """
    This sets the percentile above which precipitation values will be capped. The cap is applied to prevent
    extreme outliers from distorting the quantile mapping process.
    
    Recommendation:
    A value of 99.9 is generally a safe choice, as it allows most of the data to be preserved while still
    capping the most extreme outliers. If your analysis is very sensitive to extreme events and you want
    to be more conservative in handling them, you could lower this to 99.5 or 99.0. Conversely, if you
    want to preserve as much of the extreme data as possible, consider setting it even higher, though
    this is rare.
    """
    UPPER_CAP_THRESHOLD_PERCENTILE = 99.9
    ```

    :::

- __Deep Learning Parameters:__ Training parameters including epochs, batch size, dropout rates, filter sizes, number of filters, dense layer size, and optimizer.

    :::{admonition} Code
    :class: dropdown

    ```python
    # +++++++++++++++++++++++++++++++++++++++++
    # Configurable Parameters for Deep Learning Model
    # +++++++++++++++++++++++++++++++++++++++++
    
    """
    The DL_EPOCHS parameter controls the number of epochs (full training cycles over the dataset) during
    model training. Higher values allow the model to learn more but can lead to overfitting if trained
    for too long, especially if the dataset is small.
    Conversely, fewer epochs may result in underfitting if the model hasn't learned enough.
    
    Best Use:
    * Default (50): Provides a good balance between learning and preventing overfitting for medium-sized
                    datasets.
    * Increase: Use more epochs (e.g., 100) for larger datasets or more complex models.
    * Decrease: Use fewer epochs (e.g., 30) for smaller datasets or faster experimentation.
    """
    DL_EPOCHS = 50  # Default: 50
    
    """
    The DL_BATCH_SIZE parameter sets the number of samples used in each iteration of training.
    A larger batch size will process more samples in one step, speeding up training, but may require
    more memory and lead to less generalization.
    Smaller batch sizes take longer but often result in better generalization and performance.
    
    Best Use:
    * Default (32): A common choice that balances memory usage and training speed.
    * Increase: Use larger sizes (e.g., 64 or 128) for faster training on large datasets.
    * Decrease: Use smaller sizes (e.g., 16) to improve model generalization, especially on smaller
                datasets.
    """
    DL_BATCH_SIZE = 64  # Default: 32
    
    """
    The DL_VALIDATION_SPLIT parameter defines the fraction of data used for validation during training.
    A higher validation split gives better insight into how the model generalizes but reduces the amount
    of data available for training.
    A lower validation split increases training data, but may result in less reliable validation metrics.
    
    Best Use:
    * Default (0.2): 20% validation is typical and provides reliable feedback on model generalization.
    * Increase: Use a larger split (e.g., 0.3) for very large datasets to dedicate more data to validation.
    * Decrease: Use a smaller split (e.g., 0.1) if data is limited and you need more data for training.
    """
    DL_VALIDATION_SPLIT = 0.2  # Default: 0.2
    
    """
    The DL_EARLY_STOPPING_PATIENCE parameter defines how many epochs the model is allowed to continue
    training without improvement on the validation set.
    A higher patience value lets the model explore further before stopping, but might overfit if there
    is no improvement.
    A lower value will stop earlier, potentially missing better solutions.
    
    Best Use:
    * Default (5): Provides a balance between stopping early and giving the model enough time to improve.
    * Increase: Use a higher patience value (e.g., 10) for more complex models or noisy datasets.
    * Decrease: Use a smaller patience value (e.g., 3) for smaller datasets or faster convergence.
    """
    DL_EARLY_STOPPING_PATIENCE = 5  # Default: 5
    
    """
    The dropout parameters (DL_DROPOUT_RATE_1, DL_DROPOUT_RATE_2, DL_DROPOUT_RATE_DENSE) control the
    amount of dropout applied during training.
    Dropout randomly disables a fraction of neurons, helping to prevent overfitting by making the model
    less reliant on specific neurons.
    Higher values apply more dropout, which reduces overfitting but slows down learning.
    
    Best Use:
    * Default (0.2, 0.3, 0.4): These values provide a good balance between regularization and model
                               capacity.
    * Increase: Use higher rates (e.g., 0.5) if you observe overfitting.
    * Decrease: Use lower rates (e.g., 0.1) if you suspect underfitting or too slow learning.
    """
    DL_DROPOUT_RATE_1 = 0.2  # Dropout after the first Conv2D layer (default: 0.2)
    DL_DROPOUT_RATE_2 = 0.3  # Dropout after the second Conv2D layer (default: 0.3)
    DL_DROPOUT_RATE_DENSE = 0.4  # Dropout after the dense layer (default: 0.4)
    
    """
    The DL_FILTER_SIZE_1 and DL_FILTER_SIZE_2 parameters control the size of the convolutional
    filters in the Conv2D layers.
    Larger filters capture more spatial information but may miss fine details, while smaller
    filters capture smaller features but may miss larger patterns.
    
    Best Use:
    * Default ((3, 3)): A typical filter size that works well for many spatial datasets.
    * Increase: Use larger filters (e.g., (5, 5)) if you need to capture more spatial context.
    * Decrease: Use smaller filters (e.g., (2, 2)) if the dataset has fine details that require
                more precision.
    """
    DL_FILTER_SIZE_1 = (3, 3)  # Filter size in the first Conv2D layer (default: (3, 3))
    DL_FILTER_SIZE_2 = (3, 3)  # Filter size in the second Conv2D layer (default: (3, 3))
    
    """
    The DL_NUM_FILTERS_1 and DL_NUM_FILTERS_2 parameters define the number of filters (or feature
    detectors) in the convolutional layers.
    More filters allow the model to capture more features but increase computational cost and risk
    of overfitting. Fewer filters reduce the model's capacity but improve computational efficiency.
    
    Best Use:
    * Default (32, 64): Provides a good trade-off between learning capacity and computation.
    * Increase: Use higher numbers of filters (e.g., 64 and 128) for more complex datasets or
                larger models.
    * Decrease: Use fewer filters (e.g., 16 and 32) for smaller datasets or faster experiments.
    """
    DL_NUM_FILTERS_1 = 32  # Number of filters in the first Conv2D layer (default: 32)
    DL_NUM_FILTERS_2 = 64  # Number of filters in the second Conv2D layer (default: 64)
    
    """
    The DL_DENSE_LAYER_SIZE parameter controls the number of neurons in the fully connected (dense) layer.
    A larger size increases the model's learning capacity but also increases the risk of overfitting and
    computational complexity.
    A smaller size limits the model's ability to learn complex patterns but may reduce overfitting.
    
    Best Use:
    * Default (128): Works well for most datasets, balancing complexity and generalization.
    * Increase: Use larger sizes (e.g., 256 or 512) for more complex datasets or problems that require
                more learning capacity.
    * Decrease: Use smaller sizes (e.g., 64) for smaller datasets or simpler models.
    """
    DL_DENSE_LAYER_SIZE = 128  # Default: 128
    
    """
    The DL_OPTIMIZER parameter controls the algorithm used to update the model's weights
    during training. 'Adam' is a popular default optimizer that works well for most problems.
    Other optimizers, like SGD or RMSprop, may provide better performance for specific tasks.
    
    Best Use:
    * Default ('adam'): Works well in most scenarios.
    * Change: Try different optimizers (e.g., 'SGD' or 'RMSprop') if you want to experiment with
              different learning dynamics.
    """
    DL_OPTIMIZER = 'adam'  # Default: 'adam'
    ```

    :::

## Input/Output

The `io.py` module focuses on file input/output operations for the workflow. It provides functions to save corrected precipitation datasets, handle time aggregation, and manage NetCDF file operations.

__Key Functions:__

- `save_corrected_precip(precip_data, ds, method_abbr, method_full, folder, dekad_str, month_str)`
  Saves the corrected precipitation data to a NetCDF file following CF metadata conventions.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Helper function to save precipitation data to NetCDF with consistent metadata
  def save_corrected_precip(
          precip_data,
          ds,
          method_abbr,
          method_full,
          folder,
          dekad_str,
          month_str
      ):
      """
      Save precipitation data to NetCDF with metadata and proper filename formatting.
  
      Parameters:
      precip_data (xarray.DataArray): Precipitation data to be saved.
      ds (xarray.Dataset): Original dataset for coordinates and attributes.
      method_abbr (str): Abbreviation of the method (e.g., 'ls', 'lseqm', 'lseqmdl').
      method_full (str): Full name of the method (e.g., 'Linear Scaling', 'LSEQM').
      folder (str): Directory where the corrected precipitation will be saved.
      dekad_str (str): String representing the dekad (e.g., '01', '11', '21').
      month_str (str):  Two-digit string representing the month (e.g., '01', '02', ..., '12').
  
      Returns:
      -------
      str or None
          Path to the saved file, or None if saving failed.
      """
      # Generate output filename
      output_file = output_filename_template.format(
          folder=folder,
          method_abbr=method_abbr,
          month_str=month_str,
          dekad_str=dekad_str
      )
  
      # Check if output file exists
      if os.path.exists(output_file):
          logging.info(f"File {output_file} already exists.")
          decision = set_user_decision()  # Now returns the decision.
  
          if decision == 'S':
              logging.info(f"Skipping file {output_file}")
              return  # Skip saving
          elif decision == 'A':
              logging.info("Aborting process.")
              raise SystemExit  # Abort the process if the user decides to stop
          elif decision == 'O':
              logging.info(f"Overwriting file {output_file}")
  
      logging.info(f"Precip data dims: {precip_data.dims}, shape: {precip_data.shape}")
      # Ensure that precip_data has dimensions ('time', 'lat', 'lon')
      expected_dims = ('time', 'lat', 'lon')
  
      # If there's no 'time' dimension, create a dummy time dimension of length 1
      if 'time' not in precip_data.dims:
          logging.warning("Data has no 'time' dimension; adding dummy time dimension.")
          # Turn shape (lat, lon) -> (time, lat, lon)
          precip_data = precip_data.expand_dims(dim={'time': [pd.Timestamp.now()]}, axis=0)
  
      # Ensure lat, lon, time are in correct order
      # If user already has (time, lat, lon), no problem
      missing_dims = [d for d in expected_dims if d not in precip_data.dims]
      if missing_dims:
          logging.warning(f"Missing dims {missing_dims}, cannot reorder precisely.")
      else:
          # Reorder the dimensions
          precip_data = precip_data.transpose(*expected_dims)
  
      # Extract data from precip_data
      precip_values = precip_data.data
  
      # Create xarray Dataset for corrected precipitation
      corrected_ds = xr.Dataset(
          data_vars={
              'precipitation': (
                  ('time', 'lat', 'lon'), precip_values
              )
          },
          coords={
              'time': precip_data['time'],
              'lat': precip_data['lat'],
              'lon': precip_data['lon']
          },
          # Below information will appear as metadata in the output file
          # Feel free to adjust or modify, especially on the creator name, role and email
          attrs={
              'cdm_data_type': 'GRID',
              'title': f'Bias Corrected IMERG Late Precipitation using {method_full}',
              'summary': f'Precipitation data corrected using {method_full}',
              'source': 'IMERG and CPC-UNI',
              'history': f'Created on {pd.Timestamp.now()}',
              'DOI': '10.5067/GPM/IMERGDL/DAY/07',
              'creator_name': 'Benny Istanto',
              'creator_role': 'Climate Geographer',
              'creator_email': 'bistanto@worldbank.org',
              'comment': f'This dataset has been bias corrected using {method_full}'
          }
      )
  
      # Update metadata attributes
      corrected_ds['precipitation'].attrs.update({
          'units': 'mm',
          'long_name': 'Corrected daily mean precipitation rate estimate',
          'standard_name': 'corrected_precipitation'
      })
  
      corrected_ds['lat'].attrs.update({'units': 'degrees_north', 'long_name': 'Latitude'})
      corrected_ds['lon'].attrs.update({'units': 'degrees_east', 'long_name': 'Longitude'})
  
      # Apply land-sea mask to the `precipitation` variable only
      masked_precip = apply_land_sea_mask(corrected_ds['precipitation'], mask_file)
  
      # Replace the precipitation variable in corrected_ds with the masked version:
      corrected_ds['precipitation'] = masked_precip
  
      # Save to NetCDF following CF Convention
      try:
          corrected_ds.to_netcdf(output_file, encoding=cf18_f32, engine='netcdf4')
          logging.info(f"Saved {method_full} corrected precipitation for month {month_str}, dekad {dekad_str} at {output_file}")
          return output_file
      except IOError as e:
          logging.error(f"Failed to save file {output_file}: {str(e)}")
          return None
  ```

  :::
  
- `get_max_day_in_month(ds, month)`
  Determines the maximum day in a specified month by scanning across all years in the dataset, accounting for leap years.
  
  :::{admonition} Code
  :class: dropdown

  ```python
  # Find the maximum day in `month` across all years
  def get_max_day_in_month(
          ds: xr.Dataset,
          month: int
      ) -> int:
      """
      Scan all years in the dataset `ds`, and find the maximum day
      for the specified month. For example, if month=2 (February) and
      there's at least one leap year in ds, this returns 29; otherwise 28.
      """
      unique_years = np.unique(ds['time.year'].values)
      max_day = 0
      for year in unique_years:
          days_in_this_month = calendar.monthrange(year, month)[1]  # e.g. 28 or 29 for February
          if days_in_this_month > max_day:
              max_day = days_in_this_month
      return max_day
  ```

  :::
  
- `aggregate_data_across_years(imerg_ds, cpc_ds, month, dekad_start_day, dekad_end_day)`
  Aggregates multi-year data for a specified dekad from both the IMERG and CPC datasets.
  
  :::{admonition} Code
  :class: dropdown

  ```python
  # Aggregate IMERG and CPC data across all years for the specified dekad.
  def aggregate_data_across_years(
          imerg_ds,
          cpc_ds,
          month,
          dekad_start_day,
          dekad_end_day
      ):
      """
      Aggregate IMERG and CPC data across all years for the specified dekad.
  
      Parameters:
      imerg_ds : xarray.Dataset
          IMERG precipitation dataset with dimensions ('time', 'lat', 'lon').
      cpc_ds : xarray.Dataset
          CPC precipitation dataset with dimensions ('time', 'lat', 'lon').
      month : int
          The month number (1-12) for which the dekad is specified.
      dekad_start_day : int
          Start day of the dekad (e.g., 1, 11, 21).
      dekad_end_day : int
          End day of the dekad (e.g., 10, 20, last day of month).
  
      Returns:
      tuple of xarray.DataArray
          Aggregated IMERG and CPC data for the specified dekad across all years.
      """
      # Align datasets before applying masks
      logging.info("Aligning IMERG and CPC datasets...")
      imerg_ds, cpc_ds = xr.align(imerg_ds, cpc_ds, join="inner")
  
      # Create time masks
      logging.info("Creating time-based masks...")
      imerg_time_mask = (
          (imerg_ds['time.month'] == month) &
          (imerg_ds['time.day'] >= dekad_start_day) &
          (imerg_ds['time.day'] <= dekad_end_day)
      )
      cpc_time_mask = (
          (cpc_ds['time.month'] == month) &
          (cpc_ds['time.day'] >= dekad_start_day) &
          (cpc_ds['time.day'] <= dekad_end_day)
      )
  
      # Apply time masks
      try:
          logging.info("Applying time masks...")
          imerg_dekad_data = imerg_ds['precipitation'].where(imerg_time_mask, drop=True)
          cpc_dekad_data = cpc_ds['precip'].where(cpc_time_mask, drop=True)
      except Exception as e:
          logging.error("Error during masking:", exc_info=True)
          raise ValueError(f"Failed to apply time masks: {str(e)}")
  
      # Validate resulting data
      if imerg_dekad_data.size == 0 or cpc_dekad_data.size == 0:
          logging.error("No data available after masking.")
          logging.info(f"IMERG data after masking: {imerg_dekad_data.shape}")
          logging.info(f"CPC data after masking: {cpc_dekad_data.shape}")
          raise ValueError("No data found for the specified month and dekad.")
  
      return imerg_dekad_data, cpc_dekad_data
  ```

  :::

## Utility

The `utility.py` module provides helper functions that support data preparation and I/O management throughout the bias correction workflow.

__Key Functions:__

- `set_user_decision()`  
  Prompts the user for a decision when a file already exists (Overwrite, Skip, or Abort).

  :::{admonition} Code
  :class: dropdown

  ```python
  # User decision on existing files
  def set_user_decision():
      """
      Prompt the user for a decision when an output file already exists and return the decision.
  
      Returns:
        str: The user's decision - 'O' for Overwrite, 'S' for Skip, or 'A' for Abort.
      """
      global user_choice
      if user_choice is None:
          decision = input(
              "An output file already exists. Do you want to Overwrite (O), Skip (S), or Abort (A): "
          ).upper()
          while decision not in ['O', 'S', 'A']:
              logging.info("Invalid choice. Please choose again.")
              decision = input(
                  "Choose an action - Overwrite (O), Skip (S), Abort (A): "
              ).upper()
          user_choice = decision
      return user_choice
  ```

  :::
  
- `apply_land_sea_mask(data, mask_file)`  
  Applies a land-sea mask to the given dataset, ensuring that only terrestrial areas are processed.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Apply the mask to take out the sea
  def apply_land_sea_mask(
          data,
          mask_file
      ):
      """
      Apply the land-sea mask to the input dataset.
  
      Parameters:
      data (xarray.DataArray or xarray.Dataset): The data to which the land-sea mask should be applied.
      mask_file (str): Path to the NetCDF file containing the land-sea mask.
  
      Returns:
      xarray.DataArray or xarray.Dataset: The data with the land-sea mask applied, keeping only land areas.
      """
      # Load the land-sea mask from the external NetCDF file
      mask_ds = xr.open_dataset(mask_file)
  
      # Create a boolean mask from the 'land' variable
      land_sea_mask = mask_ds['land']
  
      # Log ranges and shapes
      logging.info(f"Data shape: {data.shape}, Mask shape: {land_sea_mask.shape}")
      logging.info(f"Data lat range: {data.lat.min().values} to {data.lat.max().values}")
      logging.info(f"Mask lat range: {land_sea_mask.lat.min().values} to {land_sea_mask.lat.max().values}")
      logging.info(f"Data lon range: {data.lon.min().values} to {data.lon.max().values}")
      logging.info(f"Mask lon range: {land_sea_mask.lon.min().values} to {land_sea_mask.lon.max().values}")
  
      # Interpolate the mask to match data resolution
      land_sea_mask_reindexed = land_sea_mask.interp(lat=data.lat, lon=data.lon, method="nearest")
  
      # Log interpolated mask shape
      logging.info(f"Reindexed mask shape: {land_sea_mask_reindexed.shape}")
  
      # Apply the mask
      masked_data = data.where(land_sea_mask_reindexed == 1, drop=True)
  
      # Log resulting data shape
      logging.info(f"Masked data shape: {masked_data.shape}")
  
      mask_ds.close()
  
      return masked_data
  ```

  :::
  
- `ensure_strict_monotonic_time(ds)`  
  Processes the time index of a dataset to enforce strict monotonicity and remove duplicates.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Ensure time index is strictly monotonic, sorted, and duplicates are removed
  def ensure_strict_monotonic_time(
          ds
      ):
      """
      Ensure the time index in the dataset is strictly monotonic, sorted, and duplicates are removed.
  
      Parameters:
      ds (xarray.Dataset): The dataset to process.
  
      Returns:
      xarray.Dataset: The dataset with a cleaned and strictly monotonic time index.
      """
      # Sort by time to ensure monotonicity
      ds = ds.sortby('time')
  
      # Remove any duplicate timestamps
      ds = ds.sel(time=~ds.get_index("time").duplicated())
  
      # Ensure strict monotonicity by dropping non-monotonic entries
      time_diff = ds['time'].diff('time')
      non_monotonic = time_diff <= pd.Timedelta(0)
  
      if non_monotonic.any():
          logging.warning(f"Found non-monotonic time steps: {ds['time'].where(non_monotonic, drop=True).values}")
          ds = ds.sel(time=~non_monotonic)
  
      return ds
  ```

  :::
  
- `reindex_and_align_with_monotonicity(reference_ds, secondary_ds, land_sea_mask)`  
  Reindexes and aligns a secondary dataset to a reference dataset (e.g., aligning CPC data with IMERG data) while ensuring a monotonic time index.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Reindex and align datasets while ensuring strict monotonicity
  def reindex_and_align_with_monotonicity(
          reference_ds,
          secondary_ds,
          land_sea_mask
      ):
      """
      Reindex and align the secondary dataset with a reference dataset while ensuring
      the reference dataset has a strictly monotonic, duplicate-free time index.
  
      Parameters
      ----------
      reference_ds : xarray.Dataset
          The reference dataset (e.g., IMERG) to which we'll align.
      secondary_ds : xarray.Dataset
          The secondary dataset (e.g., CPC) that needs alignment.
      land_sea_mask : xarray.DataArray
          Land-sea mask for spatial alignment.
  
      Returns
      -------
      tuple(xarray.Dataset, xarray.DataArray)
          The aligned secondary dataset and a spatially aligned land-sea mask.
      """
      # Ensure strict monotonicity in time for both datasets
      reference_ds = ensure_strict_monotonic_time(reference_ds)
      secondary_ds = ensure_strict_monotonic_time(secondary_ds)
  
      # Reindex and align secondary dataset to the reference dataset
      secondary_ds_aligned = secondary_ds.reindex_like(reference_ds, method='nearest')
  
      # Align land-sea mask spatially with the reference dataset
      land_sea_mask_aligned = land_sea_mask.interp(lat=reference_ds.lat, lon=reference_ds.lon, method='nearest')
  
      return secondary_ds_aligned, land_sea_mask_aligned
  ```

  :::
  
## Distribution Fitting

This `distribution_fitting.py` module is dedicated to statistical distribution fitting and quantile mapping essential for the bias correction process. It ensures that the statistical properties of the satellite and observation datasets are properly aligned.

__Key Functions:__

- `calculate_l_moments(data)`  
  Computes L-moments and L-moment ratios, providing robust measures of the distribution shape.

  :::{admonition} Code
  :class: dropdown

  ```python
  # L-Moment Calculation Function
  def calculate_l_moments(
          data
      ):
      """
      Calculate L-moments and L-moment ratios for the given data.
  
      L-moments are statistics used to describe the shape of a probability distribution.
      They are analogous to conventional moments but can be more robust to outliers.
  
      Parameters:
      data (numpy.ndarray): The input data.
  
      Returns:
      tuple: L-moments (l1, l2, l3, l4) and L-moment ratios (t2, t3, t4)
      """
      # Sort the data in ascending order
      # This is required for the calculation of probability weighted moments
      sorted_data = np.sort(data)
      n = len(data)
  
      # Calculate the first four probability weighted moments (PWMs)
      # PWMs are precursors to L-moments and are defined as:
      # β_r = E[X * F^r(X)], where F is the cumulative distribution function
  
      # b0 is simply the mean of the data
      b0 = np.mean(sorted_data)
  
      # b1, b2, b3 are calculated using discrete estimators
      # The formulas use combinatorial weights based on the sorted data
      b1 = np.sum((np.arange(1, n) * sorted_data[1:]) / (n * (n - 1)))
      b2 = np.sum((np.arange(1, n-1) * (np.arange(2, n) * sorted_data[2:])) / (n * (n - 1) * (n - 2)))
      b3 = np.sum((np.arange(1, n-2) * (np.arange(2, n-1) * (np.arange(3, n) * sorted_data[3:]))) / (n * (n - 1) * (n - 2) * (n - 3)))
  
      # Calculate L-moments
      # L-moments are linear combinations of PWMs
      l1 = b0  # L1 is the mean (measure of location)
      l2 = 2 * b1 - b0  # L2 is a measure of scale (analogous to standard deviation)
      l3 = 6 * b2 - 6 * b1 + b0  # L3 is a measure of skewness
      l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0  # L4 is a measure of kurtosis
  
      # Calculate L-moment ratios
      # These ratios are dimensionless and often more interpretable
      t2 = l2 / l1 if l1 != 0 else np.nan  # L-CV (coefficient of L-variation)
      t3 = l3 / l2 if l2 != 0 else np.nan  # L-skewness
      t4 = l4 / l2 if l2 != 0 else np.nan  # L-kurtosis
  
      return l1, l2, l3, l4, t2, t3, t4
  ```

  :::
  
- `fit_gamma_with_l_moments(data)`  
  Uses L-moments to fit a gamma distribution to the precipitation data.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Function to fit a gamma distribution using L-moments
  def fit_gamma_with_l_moments(
          data
      ):
      """
      Fit a gamma distribution to the data using L-moments.
  
      The gamma distribution is a two-parameter continuous probability distribution.
      It's often used to model positive-valued random variables.
  
      Parameters:
      data (numpy.ndarray): Array of data values.
  
      Returns:
      tuple: Fitted parameters (shape, loc, scale) of the gamma distribution.
      """
      # Remove NaN values from data
      # This is important because NaNs can affect the calculation of L-moments
      data = data[~np.isnan(data)]
  
      # If no data left after removing NaNs, return default values
      # This prevents errors in case of all-NaN input
      if len(data) == 0:
          return 1, 0, 1  # Default values: shape=1, loc=0, scale=1
  
      # Calculate L-moments
      # We only need l1 (mean), l2 (L-scale), and t2 (L-CV) for gamma fitting
      l1, l2, _, _, t2, _, _ = calculate_l_moments(data)
  
      # Estimate the gamma parameters using the L-moments
      # The shape parameter (k) is estimated using the L-CV (t2)
      # The relationship between shape and L-CV is: t2 = 1 / sqrt(k)
      shape = (2 / t2) if t2 != 0 else 0.001
      # We use a small positive value (0.001) if t2 is zero to avoid division by zero
  
      # The scale parameter (θ) is estimated using L2 and the shape parameter
      # For gamma distribution: L2 = θ * sqrt(k)
      scale = l2 / shape if shape != 0 else l2
      # If shape is zero (shouldn't happen due to the 0.001 safeguard), we use L2 as scale
  
      # Location parameter
      # For a standard gamma distribution, location is typically set to 0
      loc = 0
  
      return shape, loc, scale
  ```

  :::
  
- `fit_generalized_pareto_distribution(data, threshold)`  
  Fits a Generalized Pareto Distribution (GPD) to the excesses above a given threshold.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Fit a Generalized Pareto Distribution (GPD)
  def fit_generalized_pareto_distribution(
          data,
          threshold
      ):
      """
      Fit a Generalized Pareto Distribution (GPD) to the excesses above the threshold.
  
      The GPD is often used in extreme value theory to model the tail of a distribution.
      It's particularly useful for modeling events that exceed a high threshold.
  
      Parameters:
      data (numpy.ndarray): Array of data values.
      threshold (float): Threshold value for defining the excesses.
  
      Returns:
      tuple: Fitted parameters of the GPD (shape, location, scale).
      """
      # Calculate excesses above the threshold
      excesses = data[data > threshold] - threshold
  
      # Check if there are enough excesses for reliable fitting
      if len(excesses) < 10:  # Arbitrary minimum number of points for GPD fitting
          return (0, 0, 1)  # Return a default GPD with zero shape, zero location, and unit scale
  
      # Fit the GPD to the excesses
      # genpareto.fit returns (shape, loc, scale)
      params = genpareto.fit(excesses)
      return params
  ```

  :::
  
- `cross_validate_gpd(data, threshold, n_splits)`  
  Performs K-Fold cross-validation to yield stable GPD parameters.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Cross-validate GPD fitting
  def cross_validate_gpd(
          data,
          threshold,
          n_splits=N_SPLITS_GPD_CROSSVALIDATE
      ):
      """
      Cross-validate GPD fitting by splitting data into folds.
  
      This function uses K-Fold cross-validation to assess the stability and reliability
      of the GPD parameter estimates.
  
      Parameters:
      data (numpy.ndarray): Array of data values.
      threshold (float): Threshold value for defining the excesses.
      n_splits (int, optional): Number of cross-validation splits. Default is 5.
  
      Returns:
      tuple: Averaged parameters of the GPD from cross-validation (shape, location, scale).
      """
      # Calculate excesses above the threshold
      excesses = data[data > threshold] - threshold
  
      # If there aren't enough excesses for cross-validation, fall back to simple fitting
      if len(excesses) < n_splits:
          return fit_generalized_pareto_distribution(data, threshold)
  
      # Initialize K-Fold cross-validator
      kf = KFold(n_splits=n_splits)
      params_list = []
  
      # Perform cross-validation
      for train_index, test_index in kf.split(excesses):
          train_data, test_data = excesses[train_index], excesses[test_index]
          # Fit GPD to training data
          params = genpareto.fit(train_data)
          params_list.append(params)
  
      # Calculate average parameters across all folds
      shape_avg = np.mean([params[0] for params in params_list])
      loc_avg = np.mean([params[1] for params in params_list])
      scale_avg = np.mean([params[2] for params in params_list])
  
      return shape_avg, loc_avg, scale_avg
  ```

  :::
  
- `gamma_quantile_mapping(imerg_values, cpc_values)`  
  Performs gamma distribution-based quantile mapping with tail adjustment. This function adjusts the entire distribution of IMERG data to match CPC data, explicitly addressing extreme precipitation values using the GPD.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Gamma distribution-based quantile mapping
  def gamma_quantile_mapping(
          imerg_values,
          cpc_values
      ):
      """
      Apply gamma distribution-based quantile mapping with improved fitting and tail adjustment
      to correct the distribution of precipitation data.
  
      This function fits gamma distributions to the IMERG and CPC precipitation values using
      L-moments to improve the fitting accuracy. It then computes the cumulative distribution
      function (CDF) of the IMERG values and applies the inverse CDF of the CPC values to obtain
      the corrected precipitation values. Additionally, it adjusts the tails using the Generalized
      Pareto Distribution (GPD) to better capture extreme values.
  
      Parameters:
      imerg_values (numpy.ndarray): Array of IMERG precipitation values.
      cpc_values (numpy.ndarray): Array of CPC precipitation values.
  
      Returns:
      numpy.ndarray: Corrected precipitation values after gamma quantile mapping with improved
      fitting and tail adjustment.
      """
        # Store the original shape
      original_shape = imerg_values.shape
  
      # Flatten the arrays for processing
      imerg_flat = imerg_values.flatten()
      cpc_flat = cpc_values.flatten()
  
      # Remove NaN values from both arrays
      valid_mask = ~np.isnan(imerg_flat) & ~np.isnan(cpc_flat)
      imerg_valid = imerg_flat[valid_mask]
      cpc_valid = cpc_flat[valid_mask]
  
      # Add edge case checks here
      if imerg_valid.size == 0 or cpc_valid.size == 0:
          logging.warning("No valid data found for quantile mapping")
          return np.full(original_shape, np.nan)
  
      # Check for constant values
      if np.all(imerg_valid == imerg_valid[0]) or np.all(cpc_valid == cpc_valid[0]):
          logging.warning("Constant values detected in data")
          if np.all(imerg_valid == 0) and np.all(cpc_valid == 0):
              # If both are zero, return zeros
              return np.zeros(original_shape)
          elif np.all(imerg_valid == 0):
              # If only IMERG is zero, use CPC mean
              return np.full(original_shape, np.mean(cpc_valid))
  
      # Fit gamma distributions to the valid IMERG and CPC values
      shape1, loc1, scale1 = fit_gamma_with_l_moments(imerg_valid)
      y = gamma.cdf(imerg_valid, shape1, loc=loc1, scale=scale1)
  
      shape2, loc2, scale2 = fit_gamma_with_l_moments(cpc_valid)
      cpc_quantiles = gamma.ppf(y, shape2, loc=loc2, scale=scale2)
  
      # Ensure CPC quantiles are within realistic bounds
      cpc_quantiles = np.maximum(cpc_quantiles, 0)
  
      # Fit GPD to the tails of the CPC values with cross-validation
      threshold = np.percentile(cpc_valid, GPD_THRESHOLD_PERCENTILE)
      if not np.isnan(threshold):
          cpc_gpd_params = cross_validate_gpd(cpc_valid, threshold)
  
          # Adjust the tails using GPD if there are enough data points
          extreme_mask = imerg_valid > threshold
          if np.any(extreme_mask):
              cpc_quantiles[extreme_mask] = genpareto.ppf(
                  y[extreme_mask], *cpc_gpd_params
              ) + threshold
  
      # Dynamically determine an upper cap
      dynamic_cap = np.percentile(cpc_valid, UPPER_CAP_THRESHOLD_PERCENTILE)
      if not np.isnan(dynamic_cap):
          cpc_quantiles = np.minimum(cpc_quantiles, dynamic_cap)
  
      # Ensure non-negative corrected values
      cpc_quantiles = np.maximum(cpc_quantiles, 0)
  
      # Create an output array filled with NaNs
      corrected_values_flat = np.full_like(imerg_flat, np.nan)
  
      # Assign the corrected values back to the valid positions
      corrected_values_flat[valid_mask] = cpc_quantiles
  
      # Reshape back to the original shape
      corrected_values = corrected_values_flat.reshape(original_shape)
  
      return corrected_values
  ```

  :::
  
## Deep Learning

The `deep_learning.py` module handles all functions related to training and applying the deep learning (DL) model for bias correction. The DL component refines the EQM-corrected precipitation by further adjusting pixel-level extreme values.

__Key Functions:__

- `train_bias_correction_model(...)`  
  Trains a Convolutional Neural Network (CNN) on aggregated multi-year data (IMERG and CPC) to learn the mapping between the two datasets. This model serves as a fine-tuning step to further reduce bias, especially for extremes.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Train a deep learning model to perform bias correction
  def train_bias_correction_model(
          imerg_data,
          cpc_data,
          model_name,
          model_dir=trained_models_path,
          epochs=DL_EPOCHS,
          batch_size=DL_BATCH_SIZE,
          validation_split=DL_VALIDATION_SPLIT,
          dropout_rate_1=DL_DROPOUT_RATE_1,
          dropout_rate_2=DL_DROPOUT_RATE_2,
          dropout_rate_dense=DL_DROPOUT_RATE_DENSE,
          filter_size_1=DL_FILTER_SIZE_1,
          filter_size_2=DL_FILTER_SIZE_2,
          num_filters_1=DL_NUM_FILTERS_1,
          num_filters_2=DL_NUM_FILTERS_2,
          dense_layer_size=DL_DENSE_LAYER_SIZE,
          optimizer=DL_OPTIMIZER
      ):
      """
      Train a deep learning model to perform bias correction by learning the mapping from IMERG data to CPC data.
  
      Parameters:
      ----------
      imerg_data : xarray.DataArray
          Aggregated IMERG data for the dekad across all years.
      cpc_data : xarray.DataArray
          Aggregated CPC data for the same dekad across all years.
      model_name : str
          Name for saving the trained model.
      model_dir : str, optional
          Directory to save the trained model. Default is 'trained_models_path'.
      epochs : int, optional
          Number of training epochs.
      batch_size : int, optional
          Batch size during training.
      validation_split : float, optional
          Fraction of data for validation.
      dropout_rate_1, dropout_rate_2, dropout_rate_dense : float, optional
          Dropout rates for the network.
      filter_size_1, filter_size_2 : tuple, optional
          Convolution filter sizes.
      num_filters_1, num_filters_2 : int, optional
          Number of filters in the convolutional layers.
      dense_layer_size : int, optional
          Size of the dense layer.
      optimizer : str, optional
          Optimizer for training.
  
      Returns:
      ----------
      keras.Model
          Trained deep learning model.
      """
      # Define the path where the model will be saved
      save_path_model = os.path.join(model_dir, f"{model_name}.keras")
      logging.info(f"Checking if the model file exists at: {save_path_model}")
  
      # Check if the model file already exists
      if os.path.exists(save_path_model):
          choice = input(
              f"Model file '{save_path_model}' already exists. "
              "Use existing (U), Overwrite (O), or Abort (A)? "
          ).upper()
  
          if choice == 'U':
              # Load the existing model and return it
              logging.info(f"Using existing model: {save_path_model}")
              model = load_model(save_path_model)
              return model
  
          elif choice == 'O':
              logging.info(f"Overwriting model: {save_path_model}")
              # Proceed to train the model as usual
          else:
              logging.info("Aborting training.")
              return None
  
      # Apply land-sea mask and fill NaNs
      imerg_data = apply_land_sea_mask(imerg_data, mask_file).fillna(0)
      cpc_data = apply_land_sea_mask(cpc_data, mask_file).fillna(0)
  
      # Ensure data alignment
      imerg_data, cpc_data = xr.align(imerg_data, cpc_data)
  
      # Convert to numpy arrays
      imerg_values = imerg_data.values
      cpc_values = cpc_data.values
  
      # Reshape data for CNN input
      # Input shape: (samples, lat, lon, channels)
      # Here, samples correspond to time steps
      imerg_values = np.expand_dims(imerg_values, axis=-1)  # Shape: (samples, lat, lon, 1)
      cpc_values = np.expand_dims(cpc_values, axis=-1)      # Shape: (samples, lat, lon, 1)
  
      # Normalize the input data
      imerg_max = np.max(imerg_values)
      cpc_max = np.max(cpc_values)
      imerg_values = imerg_values / imerg_max if imerg_max != 0 else imerg_values
      cpc_values = cpc_values / cpc_max if cpc_max != 0 else cpc_values
  
      # Prepare input and output data for the model
      X = imerg_values
      y = cpc_values
  
      # Define the CNN model
      input_shape = X.shape[1:]  # Shape: (lat, lon, channels)
  
      model = Sequential([
          Input(shape=input_shape),
          Conv2D(num_filters_1, filter_size_1, activation='relu'),
          MaxPooling2D((2, 2)),
          Dropout(dropout_rate_1),
          Conv2D(num_filters_2, filter_size_2, activation='relu'),
          MaxPooling2D((2, 2)),
          Dropout(dropout_rate_2),
          Flatten(),
          Dense(dense_layer_size, activation='relu'),
          Dropout(dropout_rate_dense),
          Dense(np.prod(input_shape[:-1]), activation='linear'),
          Reshape(input_shape[:-1])
      ])
  
      # Compile the model
      model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
      model.summary(print_fn=logging.info)
  
      # Early stopping to avoid overfitting
      early_stop = EarlyStopping(
          monitor='val_loss',
          patience=DL_EARLY_STOPPING_PATIENCE,
          restore_best_weights=True  # Ensures the best weights are restored
      )
  
      # Model checkpoint to save the best model
      model_checkpoint = ModelCheckpoint(
          filepath=save_path_model,
          monitor='val_loss',
          save_best_only=True
      )
  
      # Train the model
      history = model.fit(
          X,
          y,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=validation_split,
          callbacks=[early_stop, model_checkpoint],
          verbose=1
      )
      logging.info(f"Final training history: {history.history}")
  
      # Load the best model before returning
      model = load_model(save_path_model)
  
      logging.info(f"Model training complete. Best model saved as {save_path_model}")
  
      return model
  ```

  :::
  
- `apply_deeplearning_model(model, imerg_data)`  
  Applies the trained DL model to the EQM-corrected IMERG data. Only pixels exceeding a specified threshold are replaced with DL predictions, preserving the original values for non-extreme observations.

  :::{admonition} Code
  :class: dropdown

  ```python
  # Apply the trained DL model
  def apply_deeplearning_model(
          model,
          imerg_data
      ):
      """
      Apply the trained DL model to correct IMERG data for the target dekad,
      but only overwrite pixels classified as 'extreme' based on a
      pixel-wise percentile threshold (GPD_THRESHOLD_PERCENTILE).
  
      The DL model output replaces pixels above the threshold, while
      the rest of the (non-extreme) pixels remain as the original imerg_data input,
      ensuring we keep the LSEQM (or other prior correction) for lower to moderate intensities.
  
      Parameters:
      ----------
      model : keras.Model
          Trained deep learning model for bias correction.
      imerg_data : xarray.DataArray
          Bias-corrected IMERG data for the target dekad to be refined by the model.
          Must contain a 'time' dimension.
  
      Returns:
      ----------
      xarray.DataArray
          Corrected precipitation data, where:
            - Non-extreme pixels keep imerg_data's values
            - Extreme pixels (above each pixel's threshold) are overwritten by the DL model's output
      """
      # Apply land-sea mask and fill NaNs
      imerg_data = apply_land_sea_mask(imerg_data, mask_file).fillna(0)
  
      # Add validation check
      if np.all(imerg_data == 0):
          logging.warning("All zero values in input data to DL model")
          return imerg_data
  
      # Ensure there is a 'time' dimension
      if 'time' not in imerg_data.dims:
          raise ValueError("Expected 'time' dimension in `imerg_data`, but it was not found.")
  
      # Add quality check before computing threshold
      valid_data = imerg_data.values[~np.isnan(imerg_data.values)]
      if len(valid_data) < 10:  # arbitrary minimum threshold
          logging.warning("Insufficient valid data points for reliable threshold computation")
          return imerg_data
  
      # Compute a PIXEL-WISE threshold across time dimension
      # shape => (lat, lon). E.g. 90th percentile for each pixel's distribution
      threshold_2d = imerg_data.quantile(
          q = GPD_THRESHOLD_PERCENTILE / 100.0,
          dim = 'time'  # percentile across time only
      )
  
      # Add validation for threshold values
      if np.all(np.isnan(threshold_2d)):
          logging.warning("Invalid threshold computation - all NaN values")
          return imerg_data
  
      logging.info(
          f"Pixel-wise threshold map computed at {GPD_THRESHOLD_PERCENTILE}th percentile. "
          f"threshold_2d shape: {threshold_2d.shape}"
      )
  
      corrected_slices = []
  
      # Iterate over each time step in the IMERG data, Perform DL inference on each daily 2D slice.
      # Only overwrite pixels above threshold_value.
      for t_val in imerg_data.time.values:
          # Extract a single day's data at (lat, lon)
          daily_2d = imerg_data.sel(time=t_val)
  
          # Convert daily slice into a 4D array for model inference: (batch=1, lat, lon, channels=1)
          arr_2d = daily_2d.values  # shape: (lat, lon)
          arr_2d = np.expand_dims(arr_2d, axis=-1)  # add channel dimension
          arr_2d = np.expand_dims(arr_2d, axis=0)   # add batch dimension
  
          # Normalize using max value
          arr_max = arr_2d.max()
          if arr_max != 0:
              arr_2d = arr_2d / arr_max
  
          # Model prediction for the single-day slice
          out_2d = model.predict(arr_2d)  # shape: (1, lat, lon)
          out_2d = out_2d.squeeze()       # shape: (lat, lon)
  
          # Denormalize
          out_2d = out_2d * arr_max
  
          # Overwrite only pixels above that pixel's threshold
          # Everything else remains from the original daily_2d
          final_2d = daily_2d.values.copy()  # preserve original LSEQM
          # threshold_2d might be an xarray.DataArray => align shapes if needed:
          # threshold_2d.values => shape (lat, lon)
          mask_extreme = (final_2d > threshold_2d.values)
          final_2d[mask_extreme] = out_2d[mask_extreme]
  
          # Convert back to xarray.DataArray, reintroduce 'time' dimension
          corrected_da = xr.DataArray(
              final_2d,
              dims=('lat', 'lon'),
              coords={'lat': daily_2d.lat, 'lon': daily_2d.lon},
          )
          corrected_da = corrected_da.expand_dims(dim={'time': [t_val]})
          corrected_da['time'] = [t_val]  # maintain original time coordinate
  
          corrected_slices.append(corrected_da)
  
      # Concatenate all daily slices along the time dimension
      corrected_full = xr.concat(corrected_slices, dim='time')
      # Ensure dimensions are in (time, lat, lon) order
      corrected_full = corrected_full.transpose('time', 'lat', 'lon')
  
      return corrected_full
  ```

  :::
  
## Bias Correction

This `bias_correction.py` module contains the high-level workflow function for bias correction. It orchestrates the process by combining the following steps:

- __Linear Scaling (LS):__ Adjusts the overall magnitude of the IMERG data using the ratio of CPC to IMERG means.
- __Empirical Quantile Mapping (EQM):__ Aligns the distributions via gamma quantile mapping with tail adjustment.
- __Deep Learning (DL) Enhancement:__ Optionally applies a trained DL model to further refine the corrected precipitation, particularly over extreme values.

__Key Functions:__

- `lseqmdf(...)`  
  Executes the entire bias correction process. It accepts IMERG and CPC datasets, performs aggregation, alignment, applies LS, follows with EQM (including gamma-based quantile mapping and GPD tail adjustment), and conditionally applies a DL model to correct extreme precipitation values.

  :::{admonition} Code
  :class: dropdown

  ```python
  # LSEQM+DL method for bias correction
  def lseqmdf(
          imerg_ds,
          cpc_ds,
          month,
          dekad_start_day,
          dekad_end_day,
          method_abbr="lseqm",
          method_full="Linear Scaling and Empirical Quantile Mapping",
          model=None,
          save_ls_result=True,
          save_lseqm_result=True,
          save_dl_result=True,
          month_str=None,
          dekad_str=None, 
          ls_corrected_precip_path=None, 
          lseqm_corrected_precip_path=None, 
          lseqmdl_corrected_precip_path=None
      ):
      """
      Apply Linear Scaling (LS) and Empirical Quantile Mapping (EQM) for bias correction of daily precipitation data,
      using data aggregated across years for the specified dekad. Optionally, apply a Deep Learning (DL) model for further
      correction on extreme values.
  
      Parameters:
      ----------
      imerg_ds : xarray.Dataset
          IMERG precipitation dataset with dimensions ('time', 'lat', 'lon').
      cpc_ds : xarray.Dataset
          CPC precipitation dataset with dimensions ('time', 'lat', 'lon').
      month : int
          The month number (1-12) for which the dekad is specified.
      dekad_start_day : int
          Start day of the dekad (e.g., 1, 11, 21).
      dekad_end_day : int
          End day of the dekad (e.g., 10, 20, last day of month).
      method_abbr : str, optional
          Abbreviation for the correction method (e.g., 'ls', 'lseqm'). Default is "lseqm".
      method_full : str, optional
          Full name of the correction method for file metadata. Default is "Linear Scaling and Empirical Quantile Mapping".
      model : object, optional
          Trained deep learning model for bias correction. If None, DL-based adjustments are skipped. Default is None.
      save_ls_result : bool, optional
          If True, saves the Linear Scaling (LS) corrected precipitation data. Default is True.
      save_lseqm_result : bool, optional
          If True, saves the LSEQM corrected precipitation data. Default is True.
      save_dl_result : bool, optional
          If True, saves the DL-corrected precipitation data. Default is True.
      month_str : str, optional
          Two-digit string representing the month (e.g., '01', '02', ..., '12').
      dekad_str : str, optional
          String representing the dekad (e.g., '01', '11', '21').
      ls_corrected_precip_path : str, optional
          Directory where LS-corrected data will be saved.
      lseqm_corrected_precip_path : str, optional
          Directory where LSEQM-corrected data will be saved.
      lseqmdl_corrected_precip_path : str, optional
          Directory where DL-corrected data will be saved.
  
      Returns:
      ----------
      xarray.DataArray
          The final bias-corrected precipitation data for the specified dekad.
      """
      # Ensure that month_str and dekad_str are provided
      if month_str is None or dekad_str is None:
          raise ValueError("month_str and dekad_str must be provided.")
  
      # Add data validation at the start
      if np.all(np.isnan(imerg_ds)) or np.all(np.isnan(cpc_ds)):
          logging.error("Invalid input data - all NaN values")
          raise ValueError("Invalid input data")
  
      # Aggregate data across all years for the specified dekad
      imerg_dekad_data, cpc_dekad_data = aggregate_data_across_years(imerg_ds, cpc_ds, month, dekad_start_day, dekad_end_day)
      logging.info(f"IMERG dekad data shape: {imerg_dekad_data.shape}")
      logging.info(f"CPC dekad data shape: {cpc_dekad_data.shape}")
  
      # Add validation after aggregation
      if imerg_dekad_data.size == 0 or cpc_dekad_data.size == 0:
          logging.error("No data available after aggregation")
          raise ValueError("No data available for correction")
  
      # Ensure data alignment
      imerg_dekad_data, cpc_dekad_data = xr.align(imerg_dekad_data, cpc_dekad_data, join='inner')
  
      # Perform Linear Scaling (LS)
      logging.info("Performing Linear Scaling (LS)...")
      imerg_mean = imerg_dekad_data.mean(dim='time')
      cpc_mean = cpc_dekad_data.mean(dim='time')
  
      ls_scale_factor = xr.where(
          imerg_mean != 0,
          cpc_mean / imerg_mean,
          1
      )
  
      # Apply LS scaling to IMERG data
      ls_corrected_precip = imerg_dekad_data * ls_scale_factor
  
      # Save LS result if requested
      if save_ls_result:
          logging.info("Saving LS corrected precipitation...")
          save_corrected_precip(
              ls_corrected_precip,
              imerg_dekad_data,
              method_abbr="ls",
              method_full="Linear Scaling",
              folder=ls_corrected_precip_path,
              dekad_str=dekad_str,
              month_str=month_str
          )
  
      # Perform Empirical Quantile Mapping (EQM)
      logging.info("Applying Empirical Quantile Mapping (EQM)...")
      # Apply gamma quantile mapping
      eqm_corrected_precip = xr.apply_ufunc(
          gamma_quantile_mapping,
          ls_corrected_precip,
          cpc_dekad_data,
          input_core_dims=[['time'], ['time']],
          output_core_dims=[['time']],
          vectorize=True,
          output_dtypes=[ls_corrected_precip.dtype],
          keep_attrs=True
      ).compute()
  
      # Ensure non-negative precipitation values
      eqm_corrected_precip = eqm_corrected_precip.clip(min=0)
  
      # Save LSEQM result if requested
      if save_lseqm_result:
          logging.info("Saving LSEQM corrected precipitation...")
          save_corrected_precip(
              eqm_corrected_precip,
              imerg_dekad_data,
              method_abbr="lseqm",
              method_full="Linear Scaling and Empirical Quantile Mapping",
              folder=lseqm_corrected_precip_path,
              dekad_str=dekad_str,
              month_str=month_str
          )
  
      # Apply DL model for bias correction if provided
      if model is not None:
          logging.info("Applying DL model for bias correction...")
          # Prepare input data for the model
          corrected_precip = apply_deeplearning_model(model, eqm_corrected_precip)
  
          # Ensure non-negative precipitation values
          corrected_precip = corrected_precip.clip(min=0)
  
          # Save DL corrected precipitation
          if save_dl_result:
              logging.info("Saving DL corrected precipitation...")
              save_corrected_precip(
                  corrected_precip,
                  eqm_corrected_precip,
                  method_abbr="lseqmdl",
                  method_full="Hybrid Deep Learning-Physical (Linear Scaling and Empirical Quantile Mapping) Approach",
                  folder=lseqmdl_corrected_precip_path,
                  dekad_str=dekad_str,
                  month_str=month_str
              )
  
          return corrected_precip
      else:
          return eqm_corrected_precip
  ```

  :::
  
---

This documentation chapter provides an overview of each module, its primary functions, and how these functions contribute to the overall bias correction process. Users interested in understanding the inner workings of the code can refer to the respective module files for detailed explanations and implementation details.
