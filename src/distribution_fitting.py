"""
Module: distribution_fitting.py

This module contains functions for statistical distribution fitting and quantile mapping
used in the bias correction workflow. It includes functions for:
  - Calculating L-moments and L-moment ratios (using unbiased PWM estimators from Hosking 1990).
  - Fitting a gamma distribution using MLE (scipy.stats.gamma.fit with floc=0).
  - Fitting a Generalized Pareto Distribution (GPD) to data above a threshold.
  - Performing K-Fold cross-validation to obtain stable GPD parameters.
  - Applying gamma distribution-based quantile mapping with GPD tail adjustment
    using proper conditional probability mapping for extreme values.

Dependencies:
  - Imports default parameters from the config module.
  - Uses scipy.stats and sklearn.model_selection for statistical calculations.

**Author**:
  Benny Istanto
  - Geospatial Operations Support Team, DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
  - Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.03
"""
# Import the library
import numpy as np
import xarray as xr
import logging
from scipy.stats import gamma, genpareto
from sklearn.model_selection import KFold
from .config import (N_SPLITS_GPD_CROSSVALIDATE,
                     GPD_THRESHOLD_PERCENTILE,
                     UPPER_CAP_THRESHOLD_PERCENTILE)

# +++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++

# L-Moment Calculation Function
def calculate_l_moments(
        data
    ):
    """
    Calculate L-moments and L-moment ratios for the given data.

    L-moments are statistics used to describe the shape of a probability distribution.
    They are analogous to conventional moments but can be more robust to outliers.

    Uses the unbiased PWM estimators from Hosking (1990):
        b_r = (1/n) * sum_{i=1}^{n} C(i-1,r)/C(n-1,r) * x_{i:n}
    where x_{i:n} are the order statistics.

    Parameters:
    data (numpy.ndarray): The input data.

    Returns:
    tuple: L-moments (l1, l2, l3, l4) and L-moment ratios (t2, t3, t4)
    """
    # Sort the data in ascending order
    # This is required for the calculation of probability weighted moments
    sorted_data = np.sort(data)
    n = len(data)

    if n < 4:
        logging.warning("Insufficient data points for L-moment calculation (need >= 4)")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Calculate the first four probability weighted moments (PWMs)
    # Using unbiased estimators from Hosking (1990):
    # b_r = (1/n) * sum_{i=0}^{n-1} [C(i,r) / C(n-1,r)] * x_{i:n}
    # where i is zero-based index into sorted_data

    i = np.arange(n)  # 0, 1, ..., n-1

    # b0 is simply the mean of the data
    b0 = np.mean(sorted_data)

    # b1: weights = i / (n-1)
    b1 = np.sum(i / (n - 1) * sorted_data) / n

    # b2: weights = i*(i-1) / ((n-1)*(n-2))
    b2 = np.sum(i * (i - 1) / ((n - 1) * (n - 2)) * sorted_data) / n

    # b3: weights = i*(i-1)*(i-2) / ((n-1)*(n-2)*(n-3))
    b3 = np.sum(i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)) * sorted_data) / n

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

# ----
# Function to fit a gamma distribution
def fit_gamma_distribution(
        data
    ):
    """
    Fit a gamma distribution to the data using Maximum Likelihood Estimation (MLE)
    via scipy.stats.gamma.fit with location fixed at zero.

    For precipitation data, the gamma distribution is a natural choice as it models
    positive-valued, right-skewed random variables. Fixing loc=0 ensures the distribution
    starts at zero, which is physically meaningful for precipitation.

    Parameters:
    data (numpy.ndarray): Array of data values.

    Returns:
    tuple: Fitted parameters (shape, loc, scale) of the gamma distribution.
    """
    # Remove NaN values from data
    data = data[~np.isnan(data)]

    # If no data left after removing NaNs, return default values
    if len(data) == 0:
        logging.warning("No valid data for gamma fitting, returning defaults")
        return 1, 0, 1  # Default values: shape=1, loc=0, scale=1

    # Filter to positive values only (gamma distribution requires positive data)
    positive_data = data[data > 0]

    if len(positive_data) < 5:
        logging.warning(f"Only {len(positive_data)} positive values for gamma fitting, returning defaults")
        return 1, 0, 1

    try:
        # Use scipy's MLE fitting with location fixed at 0
        # floc=0 constrains the distribution to start at zero (appropriate for precipitation)
        shape, loc, scale = gamma.fit(positive_data, floc=0)

        # Validate fitted parameters
        if shape <= 0 or scale <= 0 or np.isnan(shape) or np.isnan(scale):
            logging.warning(f"Invalid gamma parameters (shape={shape}, scale={scale}), using defaults")
            return 1, 0, 1

        return shape, loc, scale

    except Exception as e:
        logging.warning(f"Gamma fitting failed: {e}, returning defaults")
        return 1, 0, 1

# ----
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

# ----
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

    # Initialize K-Fold cross-validator with shuffle for better stability
    # Precipitation excesses have temporal ordering; shuffling ensures
    # each fold samples across the full temporal range
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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


# ----
# Fit CPC distribution parameters at native 0.5° resolution
def fit_cpc_parameters_on_native_grid(
        cpc_native_dekad
    ):
    """
    Fit gamma and GPD distribution parameters at each CPC native-resolution cell.

    This avoids redundantly fitting the same CPC time series 25 times (as happens
    when CPC is nearest-neighbour regridded to 0.1°). Instead, parameters are
    fitted once per ~0.5° cell and later interpolated to the IMERG grid.

    Based on the BCSD principle (Wood et al. 2004): correct at reference resolution,
    then disaggregate smoothly.

    Parameters
    ----------
    cpc_native_dekad : xarray.DataArray
        CPC precipitation at native ~0.5° resolution for one dekad across all years.
        Shape (n_time, n_lat_cpc, n_lon_cpc).

    Returns
    -------
    dict of xarray.DataArray
        Dictionary with 8 parameter arrays at CPC native resolution:
        'gamma_shape', 'gamma_scale', 'gpd_threshold', 'gpd_shape',
        'gpd_loc', 'gpd_scale', 'upper_cap', 'p_threshold'.
        Ocean/invalid cells are NaN.
    """
    lat_cpc = cpc_native_dekad.lat.values
    lon_cpc = cpc_native_dekad.lon.values
    n_lat = len(lat_cpc)
    n_lon = len(lon_cpc)

    # Initialize parameter arrays with NaN
    param_names = [
        'gamma_shape', 'gamma_scale', 'gpd_threshold',
        'gpd_shape', 'gpd_loc', 'gpd_scale',
        'upper_cap', 'p_threshold'
    ]
    params = {
        name: np.full((n_lat, n_lon), np.nan)
        for name in param_names
    }

    import time as _time
    _fitted = 0
    _skipped = 0
    _t0 = _time.time()

    for i in range(n_lat):
        for j in range(n_lon):
            ts = cpc_native_dekad.values[:, i, j]

            # Skip all-NaN cells (ocean)
            if np.all(np.isnan(ts)):
                _skipped += 1
                continue

            # Remove NaN and get valid data
            valid = ts[~np.isnan(ts)]
            positive = valid[valid > 0]

            # Need sufficient wet-day data for fitting
            if len(positive) < 10:
                _skipped += 1
                continue

            # Fit gamma distribution
            shape, _loc, scale = fit_gamma_distribution(valid)
            if shape <= 0 or scale <= 0:
                _skipped += 1
                continue

            params['gamma_shape'][i, j] = shape
            params['gamma_scale'][i, j] = scale

            # GPD threshold (80th percentile of all valid values)
            threshold = np.percentile(valid, GPD_THRESHOLD_PERCENTILE)
            params['gpd_threshold'][i, j] = threshold

            # Fit GPD via cross-validation
            if not np.isnan(threshold) and threshold > 0:
                gpd_shape, gpd_loc, gpd_scale = cross_validate_gpd(valid, threshold)
                params['gpd_shape'][i, j] = gpd_shape
                params['gpd_loc'][i, j] = gpd_loc
                params['gpd_scale'][i, j] = gpd_scale
            else:
                params['gpd_shape'][i, j] = 0
                params['gpd_loc'][i, j] = 0
                params['gpd_scale'][i, j] = 1

            # Upper cap (99.9th percentile)
            params['upper_cap'][i, j] = np.percentile(valid, UPPER_CAP_THRESHOLD_PERCENTILE)

            # Pre-compute CDF at threshold for conditional probability mapping
            params['p_threshold'][i, j] = gamma.cdf(threshold, shape, loc=0, scale=scale)

            _fitted += 1

        # Progress every 5 rows
        if (i + 1) % 5 == 0 or (i + 1) == n_lat:
            elapsed = _time.time() - _t0
            pct = (i + 1) / n_lat * 100
            eta = elapsed / (i + 1) * (n_lat - i - 1) if i > 0 else 0
            logging.info(
                f"  Native CPC fitting: row {i+1}/{n_lat} ({pct:.0f}%) "
                f"| {_fitted} fitted, {_skipped} skipped "
                f"| elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
            )

    logging.info(f"Fitted CPC params on native grid: {_fitted}/{n_lat * n_lon} cells")

    # Convert to xarray DataArrays
    cpc_params = {}
    for name in param_names:
        cpc_params[name] = xr.DataArray(
            params[name],
            coords={'lat': lat_cpc, 'lon': lon_cpc},
            dims=['lat', 'lon'],
            name=name
        )

    return cpc_params


# ----
# Bilinearly interpolate CPC parameters to the IMERG grid
def interpolate_cpc_params_to_imerg_grid(
        cpc_params,
        target_lat,
        target_lon
    ):
    """
    Bilinearly interpolate CPC distribution parameters from native ~0.5° to
    the IMERG 0.1° grid. Uses a two-pass approach: bilinear first, then
    nearest-neighbour to fill boundary NaN values.

    Parameters
    ----------
    cpc_params : dict of xarray.DataArray
        CPC distribution parameters at native resolution, as returned by
        fit_cpc_parameters_on_native_grid().
    target_lat : numpy.ndarray
        Target latitude values (IMERG grid).
    target_lon : numpy.ndarray
        Target longitude values (IMERG grid).

    Returns
    -------
    dict of xarray.DataArray
        Interpolated parameters at IMERG resolution.
    """
    interp_params = {}

    for name, param_da in cpc_params.items():
        # Pass 1: bilinear interpolation
        interp_da = param_da.interp(
            lat=target_lat, lon=target_lon, method='linear'
        )

        # Pass 2: fill boundary NaN with nearest-neighbour
        n_nan_before = int(interp_da.isnull().sum().item())
        if n_nan_before > 0:
            nearest_da = param_da.interp(
                lat=target_lat, lon=target_lon, method='nearest'
            )
            interp_da = interp_da.fillna(nearest_da)

        interp_params[name] = interp_da

    # Validate: clip gamma shape/scale to small positive minimum
    for key in ('gamma_shape', 'gamma_scale'):
        if key in interp_params:
            interp_params[key] = interp_params[key].clip(min=1e-6)

    # Log summary
    sample_key = 'gamma_shape'
    n_valid = int((~interp_params[sample_key].isnull()).sum().item())
    n_total = int(np.prod(interp_params[sample_key].shape))
    logging.info(
        f"Interpolated CPC params to IMERG grid: "
        f"{n_valid}/{n_total} valid pixels"
    )

    return interp_params


# ----
# Gamma quantile mapping with pre-computed CPC parameters
def gamma_quantile_mapping_precomputed(
        imerg_values,
        cpc_gamma_shape,
        cpc_gamma_scale,
        cpc_gpd_threshold,
        cpc_gpd_shape,
        cpc_gpd_loc,
        cpc_gpd_scale,
        cpc_upper_cap,
        cpc_p_threshold
    ):
    """
    Apply gamma distribution-based quantile mapping with GPD tail adjustment,
    using pre-computed CPC-side parameters instead of fitting CPC per pixel.

    This is identical to gamma_quantile_mapping() in statistical logic, but:
    - Only fits the IMERG gamma distribution (pixel-specific)
    - Uses smoothly interpolated CPC parameters (no per-pixel CPC fitting)
    - Eliminates 0.5° block boundary artefacts

    Parameters
    ----------
    imerg_values : numpy.ndarray
        IMERG precipitation time series for one pixel.
    cpc_gamma_shape : float
        Pre-computed CPC gamma shape parameter.
    cpc_gamma_scale : float
        Pre-computed CPC gamma scale parameter.
    cpc_gpd_threshold : float
        Pre-computed CPC GPD threshold (e.g. 80th percentile).
    cpc_gpd_shape : float
        Pre-computed CPC GPD shape parameter.
    cpc_gpd_loc : float
        Pre-computed CPC GPD location parameter.
    cpc_gpd_scale : float
        Pre-computed CPC GPD scale parameter.
    cpc_upper_cap : float
        Pre-computed CPC upper cap (e.g. 99.9th percentile).
    cpc_p_threshold : float
        Pre-computed CPC gamma CDF value at the GPD threshold.

    Returns
    -------
    numpy.ndarray
        Corrected precipitation values after quantile mapping.
    """
    original_shape = imerg_values.shape
    imerg_flat = imerg_values.flatten()

    # Check if any CPC parameter is NaN → skip this pixel
    if (np.isnan(cpc_gamma_shape) or np.isnan(cpc_gamma_scale) or
            np.isnan(cpc_gpd_threshold)):
        return np.full(original_shape, np.nan)

    # Remove NaN from IMERG
    valid_mask = ~np.isnan(imerg_flat)
    imerg_valid = imerg_flat[valid_mask]

    if imerg_valid.size == 0:
        return np.full(original_shape, np.nan)

    # Check for constant IMERG values
    if np.all(imerg_valid == imerg_valid[0]):
        if np.all(imerg_valid == 0):
            return np.zeros(original_shape)

    # Fit gamma to IMERG (pixel-specific)
    shape1, loc1, scale1 = fit_gamma_distribution(imerg_valid)
    y = gamma.cdf(imerg_valid, shape1, loc=loc1, scale=scale1)

    # Apply inverse CPC gamma CDF (using pre-computed CPC params)
    cpc_quantiles = gamma.ppf(y, cpc_gamma_shape, loc=0, scale=cpc_gamma_scale)
    cpc_quantiles = np.maximum(cpc_quantiles, 0)

    # GPD tail adjustment using pre-computed CPC GPD parameters
    threshold = cpc_gpd_threshold
    if not np.isnan(threshold) and threshold > 0:
        extreme_mask = imerg_valid > threshold
        if np.any(extreme_mask):
            p_threshold = cpc_p_threshold
            if not np.isnan(p_threshold) and p_threshold < 1.0:
                # Conditional probability mapping
                p_conditional = (y[extreme_mask] - p_threshold) / (1 - p_threshold)
                p_conditional = np.clip(p_conditional, 1e-10, 1 - 1e-10)

                cpc_quantiles[extreme_mask] = genpareto.ppf(
                    p_conditional, cpc_gpd_shape, loc=cpc_gpd_loc, scale=cpc_gpd_scale
                ) + threshold

    # Apply upper cap
    if not np.isnan(cpc_upper_cap):
        cpc_quantiles = np.minimum(cpc_quantiles, cpc_upper_cap)

    # Ensure non-negative
    cpc_quantiles = np.maximum(cpc_quantiles, 0)

    # Rebuild full array
    corrected_flat = np.full_like(imerg_flat, np.nan)
    corrected_flat[valid_mask] = cpc_quantiles

    return corrected_flat.reshape(original_shape)


# ----
# To address the issue of capturing extreme values in satellite data while performing EQM,
# Tail Adjustment is used to improve fit for extreme values. Tail adjustment with
# the Generalized Pareto Distribution (GPD) better captures the extreme values by specifically
# modeling the tails of the distribution, which is crucial for accurately representing
# extreme precipitation events.

# Gamma distribution-based quantile mapping
def gamma_quantile_mapping(
        imerg_values,
        cpc_values
    ):
    """
    Apply gamma distribution-based quantile mapping with tail adjustment
    to correct the distribution of precipitation data.

    This function fits gamma distributions to the IMERG and CPC precipitation values using
    MLE (scipy.stats.gamma.fit). It then computes the cumulative distribution function (CDF) of
    the IMERG values and applies the inverse CDF of the CPC values to obtain the corrected
    precipitation values. Additionally, it adjusts the tails using the Generalized Pareto
    Distribution (GPD) to better capture extreme values.

    The GPD tail adjustment uses proper conditional probability mapping:
    For values above the threshold, the unconditional CDF value is mapped to the conditional
    exceedance probability before applying the GPD inverse CDF.

    Parameters:
    imerg_values (numpy.ndarray): Array of IMERG precipitation values.
    cpc_values (numpy.ndarray): Array of CPC precipitation values.

    Returns:
    numpy.ndarray: Corrected precipitation values after gamma quantile mapping with
    tail adjustment.
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
    # (No warning — this is expected for ocean/masked pixels)
    if imerg_valid.size == 0 or cpc_valid.size == 0:
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
    shape1, loc1, scale1 = fit_gamma_distribution(imerg_valid)
    y = gamma.cdf(imerg_valid, shape1, loc=loc1, scale=scale1)

    shape2, loc2, scale2 = fit_gamma_distribution(cpc_valid)
    cpc_quantiles = gamma.ppf(y, shape2, loc=loc2, scale=scale2)

    # Ensure CPC quantiles are within realistic bounds
    cpc_quantiles = np.maximum(cpc_quantiles, 0)

    # Fit GPD to the tails of the CPC values with cross-validation
    threshold = np.percentile(cpc_valid, GPD_THRESHOLD_PERCENTILE)
    if not np.isnan(threshold) and threshold > 0:
        cpc_gpd_params = cross_validate_gpd(cpc_valid, threshold)

        # Adjust the tails using GPD with proper conditional probability mapping
        extreme_mask = imerg_valid > threshold
        if np.any(extreme_mask):
            # Compute the CDF value at the threshold under the CPC gamma distribution
            # This gives us the probability of not exceeding the threshold
            p_threshold = gamma.cdf(threshold, shape2, loc=loc2, scale=scale2)

            # Map unconditional CDF to conditional exceedance probability
            # For values above threshold: P(X > x | X > threshold) = (F(x) - F(threshold)) / (1 - F(threshold))
            # The GPD models the conditional distribution above the threshold,
            # so its ppf expects probabilities in [0, 1] representing the conditional distribution
            p_conditional = (y[extreme_mask] - p_threshold) / (1 - p_threshold)
            p_conditional = np.clip(p_conditional, 1e-10, 1 - 1e-10)  # Avoid boundary issues

            cpc_quantiles[extreme_mask] = genpareto.ppf(
                p_conditional, *cpc_gpd_params
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
