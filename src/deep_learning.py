"""
Module: deep_learning.py

This module contains functions related to training and applying the deep learning model
for bias correction. It includes:
  - train_bias_correction_model: Trains a CNN model to learn the spatial mapping from
    LSEQM-corrected data to CPC data.
  - apply_deeplearning_model: Applies the trained CNN to refine extreme pixels in the
    LSEQM result using alpha-blending.

Two-step workflow (eliminates domain shift):
  Step 1 (bias_correction.py): LS + EQM + GPD produces LSEQM-corrected data.
  Step 2 (this module):
    - Training: X = LSEQM-corrected, y = CPC (both from the same dekad aggregation).
      The model learns what LSEQM missed — residual spatial patterns, extreme refinement.
    - Inference: the trained model receives LSEQM-corrected data (same domain as training),
      and the prediction is alpha-blended with the LSEQM result for extreme pixels only.

Design rationale (Sha et al. 2020; Pan et al. 2019):
  The physical-statistical pipeline (LS + EQM + GPD) handles the bulk of the distributional
  correction and should be trusted. The DL model learns residual spatial patterns that
  the pixel-wise statistical methods cannot capture (e.g. orographic effects, convective
  organisation). At inference the DL prediction is **blended** with the LSEQM output
  rather than replacing it, so that:
    - Extremes are never crushed (LSEQM carries most of the weight via DL_BLEND_ALPHA)
    - Zero rain stays zero
    - Spatial detail from IMERG is preserved

The module imports configuration parameters from config.py.

**Author**:
  Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""
# Import the library
import os
from contextlib import nullcontext as _nullcontext
import numpy as np
import xarray as xr
import logging

# TensorFlow / Keras imports.
# Wrapped in try/except so that the module can be *imported* in environments
# where TensorFlow is not installed (e.g. lightweight metric/QA workflows).
# A clear error is raised only when a function that actually needs TF is called.
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model, Sequential  # type: ignore
    from tensorflow.keras.layers import (  # type: ignore
        Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
    _HAS_TENSORFLOW = True

    # --- GPU memory growth ---------------------------------------------------
    # By default TensorFlow pre-allocates ALL GPU memory, which can cause
    # out-of-memory errors when running alongside other GPU processes (e.g.
    # the display server).  Setting memory_growth=True tells TF to allocate
    # only as much GPU memory as needed, growing incrementally.
    # This is safe and recommended for interactive / notebook workflows.
    try:
        for _gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(_gpu, True)
    except (RuntimeError, ValueError) as _e:
        # Memory growth must be set before GPUs are initialized.
        # If it's too late (e.g. module re-imported), log and continue.
        logging.debug(f"Could not set GPU memory growth: {_e}")

except ImportError:
    _HAS_TENSORFLOW = False
    logging.info(
        "TensorFlow not found. Deep learning functions will not be available. "
        "Install with: pip install tensorflow"
    )

from .config import (DL_EPOCHS, DL_BATCH_SIZE, DL_VALIDATION_SPLIT,
                    DL_EARLY_STOPPING_PATIENCE, DL_DROPOUT_RATE_1, DL_DROPOUT_RATE_2,
                    DL_DROPOUT_RATE_DENSE, DL_FILTER_SIZE_1, DL_FILTER_SIZE_2,
                    DL_NUM_FILTERS_1, DL_NUM_FILTERS_2, DL_DENSE_LAYER_SIZE, DL_OPTIMIZER,
                    DL_BLEND_ALPHA, GPD_THRESHOLD_PERCENTILE)


def _require_tensorflow():
    """Raise a clear ImportError if TensorFlow is not installed."""
    if not _HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for deep learning functions but is not installed. "
            "Install it with:\n"
            "  pip install tensorflow          # CPU-only\n"
            "  pip install tensorflow[and-cuda] # GPU support (NVIDIA)\n"
            "See https://www.tensorflow.org/install for details."
        )

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# ----
# Train a deep learning model to perform bias correction
def train_bias_correction_model(
        input_data,
        target_data,
        model_name,
        mask_data=None,
        model_dir=None,
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
        optimizer=DL_OPTIMIZER,
        interactive=True
    ):
    """
    Train a deep learning model to perform bias correction by learning the
    spatial mapping from LSEQM-corrected data to CPC data.

    In the two-step workflow, this function receives LSEQM-corrected data
    (not raw IMERG) as input, eliminating the domain shift between training
    and inference. The model learns what the physical-statistical correction
    missed — residual spatial patterns, extreme event refinement.

    Architecture
    ------------
    A two-layer CNN with ``padding='same'`` so that spatial dimensions are
    preserved through the convolutional layers. This gives the model more
    spatial context before the Flatten -> Dense bottleneck, improving its
    ability to learn fine-grained spatial correction patterns (e.g. orographic
    enhancement, rain-shadow effects).

    Normalization
    -------------
    During training, both input and target are normalized **per-sample** (each
    daily 2-D field divided by its own maximum value).  This teaches the model
    to learn **relative spatial patterns** rather than absolute intensities.

    Parameters:
    ----------
    input_data : xarray.DataArray
        LSEQM-corrected data for the dekad across all years.
        Should already be masked (NaN over ocean).
    target_data : xarray.DataArray
        CPC (gauge-based reference) data for the same dekad across all years.
        Should already be masked (NaN over ocean).
    model_name : str
        Name for saving the trained model.
    mask_data : xarray.DataArray, optional
        Pre-loaded land-sea mask. If None, masking should have been applied upstream.
    model_dir : str, optional
        Directory to save the trained model. If None, uses config default.
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
    interactive : bool, optional
        If True, prompt user for decisions on existing models. If False, use existing model
        if available. Default is True.

    Returns:
    ----------
    keras.Model
        Trained deep learning model.
    """
    # Ensure TensorFlow is available before proceeding
    _require_tensorflow()

    # Use config default if model_dir not provided
    if model_dir is None:
        from .config import trained_models_path
        model_dir = trained_models_path

    # Define the path where the model will be saved
    save_path_model = os.path.join(model_dir, f"{model_name}.keras")
    logging.info(f"Checking if the model file exists at: {save_path_model}")

    # Check if the model file already exists
    if os.path.exists(save_path_model):
        if interactive:
            choice = input(
                f"Model file '{save_path_model}' already exists. "
                "Use existing (U), Overwrite (O), or Abort (A)? "
            ).upper()
        else:
            choice = 'U'  # Non-interactive: use existing model
            logging.info("Non-interactive mode: using existing model")

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

    # Fill NaN values with 0 for CNN processing
    # (NaN/ocean pixels should already be masked upstream; here we ensure no NaN for TensorFlow)
    input_data = input_data.fillna(0)
    target_data = target_data.fillna(0)

    # Ensure data alignment
    input_data, target_data = xr.align(input_data, target_data)

    # Convert to numpy arrays
    input_values = input_data.values  # (time, lat, lon)
    target_values = target_data.values      # (time, lat, lon)

    # --- Per-sample normalization ---
    # Each daily field is normalized by its own max.  This removes the absolute
    # intensity from the training signal and makes the model focus on the
    # *spatial distribution* of rainfall.  At inference the same per-sample
    # norm + denorm preserves the LSEQM magnitude.
    n_samples = input_values.shape[0]
    for i in range(n_samples):
        imax = input_values[i].max()
        cmax = target_values[i].max()
        if imax > 0:
            input_values[i] = input_values[i] / imax
        if cmax > 0:
            target_values[i] = target_values[i] / cmax

    # Reshape data for CNN input
    # Input shape: (samples, lat, lon, channels)
    X = np.expand_dims(input_values, axis=-1)  # Shape: (samples, lat, lon, 1)
    y = np.expand_dims(target_values, axis=-1)    # Shape: (samples, lat, lon, 1)

    # Define the CNN model
    input_shape = X.shape[1:]  # Shape: (lat, lon, channels)

    # --- Adaptive batch size --------------------------------------------------
    # Large spatial grids (e.g. 171x461 = 78 831 pixels) combined with Conv2D
    # filters need substantial GPU memory per sample.  If the user-provided
    # batch_size would require more than roughly 2 GB per batch (heuristic),
    # reduce it automatically so training fits on modest GPUs (4-6 GB VRAM).
    spatial_pixels = int(np.prod(input_shape[:-1]))
    # Rough estimate: each sample needs ~4 bytes * pixels * filters * 8 (fwd+bwd)
    est_bytes_per_sample = spatial_pixels * max(num_filters_1, num_filters_2) * 4 * 8
    max_batch_mem = 1.5 * 1024**3  # 1.5 GB target per batch
    safe_batch = max(1, int(max_batch_mem / est_bytes_per_sample))
    if safe_batch < batch_size:
        logging.info(
            f"Reducing batch_size from {batch_size} to {safe_batch} "
            f"to fit {spatial_pixels:,} pixels/sample on available GPU memory."
        )
        batch_size = safe_batch

    # Build CNN model.
    # Uses padding='same' so spatial dimensions are preserved through conv layers.
    # This gives the model more spatial context — important for capturing spatial
    # correction patterns (orographic enhancement, convective clusters) rather
    # than just learning a smooth, low-frequency field through a tiny bottleneck.
    def _build_model():
        """Build and compile the CNN model."""
        m = Sequential([
            Input(shape=input_shape),
            Conv2D(num_filters_1, filter_size_1, activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate_1),
            Conv2D(num_filters_2, filter_size_2, activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate_2),
            Flatten(),
            Dense(dense_layer_size, activation='relu'),
            Dropout(dropout_rate_dense),
            Dense(int(np.prod(input_shape[:-1])), activation='linear'),
            Reshape(input_shape[:-1])
        ])
        m.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return m

    model = _build_model()
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

    # --- Train with GPU -> CPU fallback ----------------------------------------
    # If the GPU runs out of memory (OOM) during training, we automatically
    # retry on CPU.  This is common on GPUs with <= 4 GB VRAM when processing
    # large spatial grids.
    def _train(device=None):
        """Run model.fit, optionally forcing a specific device."""
        ctx = tf.device(device) if device else _nullcontext()
        with ctx:
            return model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, model_checkpoint],
                verbose=1,
            )

    try:
        history = _train()  # Try default device (GPU if available)
    except (tf.errors.ResourceExhaustedError, tf.errors.UnknownError) as gpu_err:
        logging.warning(
            f"GPU training failed ({type(gpu_err).__name__}). "
            "Retrying on CPU -- this will be slower but avoids the OOM error."
        )
        # Rebuild model on CPU (GPU state may be corrupted after OOM)
        with tf.device('/CPU:0'):
            model = _build_model()
        history = _train(device='/CPU:0')

    logging.info(f"Final training history: {history.history}")

    # Load the best model before returning
    model = load_model(save_path_model)

    logging.info(f"Model training complete. Best model saved as {save_path_model}")

    return model

# ----
# Apply the trained DL model
def apply_deeplearning_model(
        model,
        lseqm_data,
        blend_alpha=DL_BLEND_ALPHA,
        confidence_mask=None
    ):
    """
    Apply the trained DL model to refine extreme pixels in the LSEQM-corrected data
    using **alpha-blending** rather than hard replacement.

    For each daily 2-D field the function:

    1. Computes a pixel-wise extreme threshold (GPD_THRESHOLD_PERCENTILE across time).
    2. Runs the CNN on the per-sample-normalized field.
    3. For every extreme pixel (above the threshold), blends the LSEQM value with the
       DL prediction::

           final = alpha * lseqm + (1 - alpha) * dl_predicted

       where ``alpha = DL_BLEND_ALPHA`` (default 0.7).

    4. Non-extreme pixels (below threshold) keep their LSEQM value unchanged.
    5. Zero-rain pixels are never modified (``if LSEQM == 0 -> final == 0``).

    This ensures the physical-statistical result dominates while the DL provides a
    moderate spatial refinement for extremes — matching the design goal that DL
    "fills the gap" rather than replacing physics.

    References
    ----------
    - Sha et al. (2020), Geophys. Res. Lett. — residual DL correction after statistical model
    - Pan et al. (2019), Water Resources Research — CNN residual correction for precipitation

    Parameters:
    ----------
    model : keras.Model
        Trained deep learning model for bias correction.
    lseqm_data : xarray.DataArray
        Bias-corrected LSEQM data for the target dekad.
        Must contain a 'time' dimension. Should already be masked (NaN over ocean).
    blend_alpha : float, optional
        Blending weight for LSEQM.  ``1.0`` = pure LSEQM, ``0.0`` = pure DL.
        Default is ``DL_BLEND_ALPHA`` from config (typically 0.7).
    confidence_mask : xarray.DataArray, optional
        2-D (lat, lon) confidence mask with values in [0, 1] representing
        how reliable CPC-UNI is at each grid cell based on gauge station density.
        If provided, the blending alpha is spatially modulated::

            effective_alpha = 1.0 - confidence * (1.0 - blend_alpha)

        High confidence (many stations) → effective_alpha = blend_alpha (DL active).
        Zero confidence (no stations) → effective_alpha = 1.0 (pure LSEQM).
        If None, uniform blend_alpha is used everywhere (backward compatible).

    Returns:
    ----------
    xarray.DataArray
        Corrected precipitation data, where:
          - Non-extreme pixels keep LSEQM values exactly
          - Extreme pixels are alpha-blended between LSEQM and DL prediction
          - Zero-rain pixels remain zero
    """
    # Ensure TensorFlow is available before proceeding
    _require_tensorflow()

    # Fill NaN with 0 for CNN processing (ocean pixels are NaN from upstream masking)
    lseqm_data_filled = lseqm_data.fillna(0)

    # Add validation check
    if np.all(lseqm_data_filled == 0):
        logging.warning("All zero values in input data to DL model")
        return lseqm_data

    # Ensure there is a 'time' dimension
    if 'time' not in lseqm_data.dims:
        raise ValueError("Expected 'time' dimension in `lseqm_data`, but it was not found.")

    # Add quality check before computing threshold
    valid_data = lseqm_data.values[~np.isnan(lseqm_data.values)]
    if len(valid_data) < 10:  # arbitrary minimum threshold
        logging.warning("Insufficient valid data points for reliable threshold computation")
        return lseqm_data

    # Compute a PIXEL-WISE threshold across time dimension
    # shape => (lat, lon). E.g. 80th percentile for each pixel's distribution
    # Suppress "All-NaN slice encountered" — expected for ocean/masked pixels.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        threshold_2d = lseqm_data.quantile(
            q = GPD_THRESHOLD_PERCENTILE / 100.0,
            dim = 'time'  # percentile across time only
        )

    # Add validation for threshold values
    if np.all(np.isnan(threshold_2d)):
        logging.warning("Invalid threshold computation - all NaN values")
        return lseqm_data

    logging.info(
        f"Pixel-wise threshold map computed at {GPD_THRESHOLD_PERCENTILE}th percentile. "
        f"threshold_2d shape: {threshold_2d.shape}"
    )
    logging.info(
        f"DL blending: base alpha={blend_alpha:.2f} (LSEQM weight), "
        f"1-alpha={1-blend_alpha:.2f} (DL weight)"
    )

    if confidence_mask is not None:
        logging.info(
            f"Confidence mask active: spatial alpha modulation enabled. "
            f"Confidence range: [{float(confidence_mask.min()):.3f}, "
            f"{float(confidence_mask.max()):.3f}]"
        )
    else:
        logging.info("No confidence mask: using uniform alpha everywhere")

    corrected_slices = []
    _n_extreme_total = 0
    _n_days = 0

    # Iterate over each time step in the LSEQM data
    for t_val in lseqm_data.time.values:
        _n_days += 1
        # Extract a single day's data at (lat, lon)
        daily_2d = lseqm_data.sel(time=t_val)
        daily_2d_filled = daily_2d.fillna(0)

        # Convert daily slice into a 4D array for model inference: (batch=1, lat, lon, channels=1)
        arr_2d = daily_2d_filled.values.copy()  # shape: (lat, lon)
        arr_2d = np.expand_dims(arr_2d, axis=-1)  # add channel dimension
        arr_2d = np.expand_dims(arr_2d, axis=0)   # add batch dimension

        # Per-sample normalization: normalize by THIS daily slice's max value.
        # This is consistent with how training data was normalized — each sample
        # by its own max — so the model sees the same [0,1] range it was trained on.
        arr_max = arr_2d.max()
        if arr_max != 0:
            arr_2d_norm = arr_2d / arr_max
        else:
            arr_2d_norm = arr_2d

        # Model prediction for the single-day slice
        out_2d = model.predict(arr_2d_norm, verbose=0)  # shape: (1, lat, lon)
        out_2d = out_2d.squeeze()       # shape: (lat, lon)

        # Denormalize by the SAME max value used for normalization
        out_2d = out_2d * arr_max

        # Ensure non-negative DL prediction
        out_2d = np.maximum(out_2d, 0)

        # --- Alpha-blending for extreme pixels ---
        # Start with the LSEQM result (including NaN for ocean)
        lseqm_2d = daily_2d.values.copy()
        final_2d = lseqm_2d.copy()

        # Identify extreme pixels: above the threshold AND non-zero
        lseqm_filled = daily_2d_filled.values
        mask_extreme = (lseqm_filled > threshold_2d.values) & (lseqm_filled > 0)
        _n_extreme_total += np.sum(mask_extreme)

        # Blend: final = effective_alpha * LSEQM + (1 - effective_alpha) * DL_predicted
        # This keeps LSEQM as the dominant component while allowing DL to
        # provide a spatial refinement for extremes.
        if np.any(mask_extreme):
            if confidence_mask is not None:
                # Spatially modulated alpha based on station density confidence
                conf_2d = confidence_mask.values  # shape: (lat, lon)
                effective_alpha_2d = 1.0 - conf_2d * (1.0 - blend_alpha)
                alpha_extreme = effective_alpha_2d[mask_extreme]
            else:
                # Uniform alpha (backward compatible)
                alpha_extreme = blend_alpha

            final_2d[mask_extreme] = (
                alpha_extreme * lseqm_filled[mask_extreme]
                + (1.0 - alpha_extreme) * out_2d[mask_extreme]
            )

        # Convert back to xarray.DataArray, reintroduce 'time' dimension
        corrected_da = xr.DataArray(
            final_2d,
            dims=('lat', 'lon'),
            coords={'lat': daily_2d.lat, 'lon': daily_2d.lon},
        )
        corrected_da = corrected_da.expand_dims(dim={'time': [t_val]})
        corrected_da['time'] = [t_val]  # maintain original time coordinate

        corrected_slices.append(corrected_da)

    logging.info(
        f"DL blending complete: {_n_days} daily slices processed, "
        f"{_n_extreme_total:,} extreme pixels blended (alpha={blend_alpha:.2f})"
    )

    # Concatenate all daily slices along the time dimension
    corrected_full = xr.concat(corrected_slices, dim='time')
    # Ensure dimensions are in (time, lat, lon) order
    corrected_full = corrected_full.transpose('time', 'lat', 'lon')

    return corrected_full
