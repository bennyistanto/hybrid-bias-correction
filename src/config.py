"""
Module: config.py

This module defines the default configuration for the Hybrid Bias Correction (LSEQM+DL) workflow.
It centralizes all configurable parameters and file paths so that users can override them in a
notebook or via function arguments. The configuration includes:

- Directory settings for the main project, input data, and output locations.
- Default input file paths for IMERG, CPC, and the land-sea mask.
- Output directories for storing various corrected precipitation products and trained models.
  These directories are automatically created if they do not exist.
- A centralized output filename template (OUTPUT_FILENAME_TEMPLATE) for generating standardized
  NetCDF filenames. The template follows this convention:
    idn_cli_{method_abbr}_corrected_imergl_month{month_str}_dekad{dekad_str}.nc4
  where:
    • method_abbr: 'ls' for Linear Scaling, 'lseqm' for LS+EQM, and 'lseqmdl' for the hybrid DL approach.
    • month_str: The month number as a two-digit string.
    • dekad_str: The dekad as a string (e.g., "01", "11", "21").
- CF-compliant encoding settings (cf18_f32) for NetCDF outputs.
- Statistical fitting and quantile mapping parameters:
    • N_SPLITS_GPD_CROSSVALIDATE: Number of splits for cross-validation during GPD fitting.
    • GPD_THRESHOLD_PERCENTILE: Percentile threshold used for fitting the Generalized Pareto Distribution,
      influencing the tail adjustment in the quantile mapping.
    • UPPER_CAP_THRESHOLD_PERCENTILE: Percentile used to cap extreme precipitation values.
- Deep learning model training parameters, including:
    • DL_EPOCHS, DL_BATCH_SIZE, DL_VALIDATION_SPLIT, DL_EARLY_STOPPING_PATIENCE,
    • Dropout rates (DL_DROPOUT_RATE_1, DL_DROPOUT_RATE_2, DL_DROPOUT_RATE_DENSE),
    • Convolution filter sizes (DL_FILTER_SIZE_1, DL_FILTER_SIZE_2),
    • Number of filters (DL_NUM_FILTERS_1, DL_NUM_FILTERS_2), DL_DENSE_LAYER_SIZE,
    • DL_OPTIMIZER.

These default settings can be overridden by user-specified parameters in the notebook or via function
arguments in the main execution module.

Author:
  Benny Istanto
  - GOST/DECSC/DEC Data Group, The World Bank, United States (bistanto@worldbank.org)
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia (bistanto@ipb.ac.id)

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import os
import numpy as np

# +++++++++++++++++++++++++++++++++++++++++
# Configurable Directory
# +++++++++++++++++++++++++++++++++++++++++

# Default directories (these can be overridden)
main_dir = f'/content/drive/MyDrive/exercises/gfm1609'
input_dir = f'{main_dir}/data/bc/input'
output_dir = f'{main_dir}/data/bc/output'

# Default input file paths
imerg_file = f'{input_dir}/imergl/idn_imergl.nc4' # Single daily timeseries multi-year file
cpc_file = f'{input_dir}/cpcuni/idn_cpcuni.nc4' # Single daily timeseries multi-year file
mask_file = f'{main_dir}/data/subset/iso3/idn_subset.nc' # Mask file

# Output directories for different correction products
ls_corrected_precip_path = f'{output_dir}/corrected_ls' # Directory for storing corrected precip
lseqm_corrected_precip_path = f'{output_dir}/corrected_lseqm' # Directory for storing corrected precip
lseqmdl_corrected_precip_path = f'{output_dir}/corrected_lseqmdl' # Directory for storing corrected precip
trained_models_path = f'{output_dir}/trained_models' # Directory for storing trained models

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

# List of output directories
output_directories = [
    output_dir,
    ls_corrected_precip_path,
    lseqm_corrected_precip_path,
    lseqmdl_corrected_precip_path,
    trained_models_path
]

# Create the output directories if they don't already exist
for directory in output_directories:
    os.makedirs(directory, exist_ok=True)

# The NetCDF output will follow NetCDF Climate and Forecast (CF) Metadata Convention
# https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html
# Encoding for CF 1.8 (adjust based on your precision and range requirements)
cf18_f32 = {'precipitation': {'dtype': 'float32', 'zlib': True, '_FillValue': np.nan}}

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