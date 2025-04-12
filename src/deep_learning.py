"""
Module: deep_learning.py

This module contains functions related to training and applying the deep learning model
for bias correction. It includes:
  - train_bias_correction_model: Trains a CNN model to learn the mapping from IMERG to CPC data.
  - apply_deeplearning_model: Applies the trained CNN model to adjust only those pixels classified
    as 'extreme' based on a pixel-wise threshold.

The module imports configuration parameters from config.py and uses the land-sea mask from config.py.

**Author**:
  Benny Istanto
  - GOST/DECSC/DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import os
import numpy as np
import xarray as xr
import logging
from tensorflow.keras.models import load_model, Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from .config import (DL_EPOCHS, DL_BATCH_SIZE, DL_VALIDATION_SPLIT,
                    DL_EARLY_STOPPING_PATIENCE, DL_DROPOUT_RATE_1, DL_DROPOUT_RATE_2,
                    DL_DROPOUT_RATE_DENSE, DL_FILTER_SIZE_1, DL_FILTER_SIZE_2,
                    DL_NUM_FILTERS_1, DL_NUM_FILTERS_2, DL_DENSE_LAYER_SIZE, DL_OPTIMIZER,
                    GPD_THRESHOLD_PERCENTILE, trained_models_path, mask_file)
from .utility import apply_land_sea_mask

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++
# ----
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

# ----
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
