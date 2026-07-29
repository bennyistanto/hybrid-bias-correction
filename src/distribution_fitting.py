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
  Applied Climatology Study Program, Department of Geophysics and Meteorology,
  Bogor Agricultural University, Indonesia
  Email: bennyistanto@apps.ipb.ac.id

  with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2026.07
"""
# Import the library
import numpy as np
import xarray as xr
import logging
from scipy.stats import gamma, genpareto
from sklearn.model_selection import KFold
from .config import (N_SPLITS_GPD_CROSSVALIDATE,
                     GPD_THRESHOLD_PERCENTILE,
                     UPPER_CAP_THRESHOLD_PERCENTILE,
                     WET_DAY_THRESHOLD)

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
    # Fix A (Coles 2001): pin location to 0 because we are fitting excesses
    # that are by construction non-negative. Allowing a free location can
    # drift and produces a small systematic bias in the upper tail.
    try:
        params = genpareto.fit(excesses, floc=0)
    except Exception:
        return (0.0, 0.0, 1.0)
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
        # Fit GPD to training data with floc=0 (Fix A, Coles 2001)
        try:
            params = genpareto.fit(train_data, floc=0)
            params_list.append(params)
        except Exception:
            continue

    if not params_list:
        return (0.0, 0.0, 1.0)

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
        Dictionary with 9 parameter arrays at CPC native resolution:
        'gamma_shape', 'gamma_scale', 'gpd_threshold', 'gpd_shape',
        'gpd_loc', 'gpd_scale', 'upper_cap', 'p_threshold', 'p_dry_cpc'.
        Ocean/invalid cells are NaN.

    Notes
    -----
    Fixes applied (2026.04):
      * Fix C (Cannon 2015 §3.2): gamma distribution fitted on WET-day values
        only (> WET_DAY_THRESHOLD), not on all values including zeros. The
        GPD threshold is also computed from wet values only.
      * Fix A (Coles 2001): GPD fitting inside cross_validate_gpd uses
        floc=0 (see fit_generalized_pareto_distribution).
      * New: p_dry_cpc is returned per cell so the downstream mapping
        step can apply Cannon dry-day handling using the interpolated
        dry-day frequency at each IMERG pixel.
    """
    lat_cpc = cpc_native_dekad.lat.values
    lon_cpc = cpc_native_dekad.lon.values
    n_lat = len(lat_cpc)
    n_lon = len(lon_cpc)

    # Initialize parameter arrays with NaN
    param_names = [
        'gamma_shape', 'gamma_scale', 'gpd_threshold',
        'gpd_shape', 'gpd_loc', 'gpd_scale',
        'upper_cap', 'p_threshold', 'p_dry_cpc'
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

            # Remove NaN and split into wet-day sample (Fix C, Cannon 2015)
            valid = ts[~np.isnan(ts)]
            wet = valid[valid > WET_DAY_THRESHOLD]

            # Need sufficient wet-day data for fitting
            if len(wet) < 10:
                _skipped += 1
                continue

            # Dry-day fraction (new: required by downstream Cannon mapping)
            params['p_dry_cpc'][i, j] = 1.0 - (len(wet) / len(valid))

            # Fit gamma on wet-day sample only
            shape, _loc, scale = fit_gamma_distribution(wet)
            if shape <= 0 or scale <= 0:
                _skipped += 1
                continue

            params['gamma_shape'][i, j] = shape
            params['gamma_scale'][i, j] = scale

            # GPD threshold (80th percentile of WET values, Fix C)
            threshold = np.percentile(wet, GPD_THRESHOLD_PERCENTILE)
            params['gpd_threshold'][i, j] = threshold

            # Fit GPD via cross-validation on the wet sample (Fix A inside)
            if not np.isnan(threshold) and threshold > 0:
                gpd_shape, gpd_loc, gpd_scale = cross_validate_gpd(wet, threshold)
                params['gpd_shape'][i, j] = gpd_shape
                params['gpd_loc'][i, j] = gpd_loc
                params['gpd_scale'][i, j] = gpd_scale
            else:
                params['gpd_shape'][i, j] = 0
                params['gpd_loc'][i, j] = 0
                params['gpd_scale'][i, j] = 1

            # Upper cap (99.9th percentile of wet values)
            params['upper_cap'][i, j] = np.percentile(wet, UPPER_CAP_THRESHOLD_PERCENTILE)

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
        # Pass 1: bilinear interpolation (NaN outside the convex hull of
        # CPC-native cell centres; bites small AOIs where the AOI extent
        # reaches outside the centres, e.g. Bali on a 2 x 4 CPC tile).
        interp_da = param_da.interp(
            lat=target_lat, lon=target_lon, method='linear'
        )

        # Pass 2: fill boundary NaN with true unrestricted nearest-neighbour.
        # Note: interp(method='nearest') ALSO returns NaN beyond the convex
        # hull (it goes through scipy.interpolate with bounds_error=False),
        # which silently broke small AOIs. reindex(method='nearest') is a
        # pure lookup with no convex-hull restriction.
        n_nan_before = int(interp_da.isnull().sum().item())
        if n_nan_before > 0:
            nearest_da = param_da.reindex(
                lat=target_lat, lon=target_lon, method='nearest'
            )
            interp_da = interp_da.fillna(nearest_da)

        interp_params[name] = interp_da

    # Validate: clip gamma shape/scale to small positive minimum
    for key in ('gamma_shape', 'gamma_scale'):
        if key in interp_params:
            interp_params[key] = interp_params[key].clip(min=1e-6)

    # Clip p_dry_cpc to [0, 1] - bilinear interpolation of a probability
    # can numerically drift slightly outside the valid range.
    if 'p_dry_cpc' in interp_params:
        interp_params['p_dry_cpc'] = interp_params['p_dry_cpc'].clip(min=0.0, max=1.0)

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
        cpc_p_threshold,
        p_dry_cpc
    ):
    """
    Apply gamma distribution-based quantile mapping with GPD tail adjustment,
    using pre-computed CPC-side parameters instead of fitting CPC per pixel.

    This is the BCSD-path counterpart to gamma_quantile_mapping(). Statistical
    logic is identical, but CPC parameters come from the native-resolution fit
    (fit_cpc_parameters_on_native_grid) and are bilinearly interpolated to the
    IMERG grid, eliminating 0.5° block artefacts.

    Parameters
    ----------
    imerg_values : numpy.ndarray
        IMERG precipitation time series for one pixel.
    cpc_gamma_shape : float
        Pre-computed CPC gamma shape (fitted on wet values only, Fix C).
    cpc_gamma_scale : float
        Pre-computed CPC gamma scale (fitted on wet values only, Fix C).
    cpc_gpd_threshold : float
        Pre-computed CPC GPD threshold (80th percentile of wet values).
    cpc_gpd_shape : float
        Pre-computed CPC GPD shape (fitted with floc=0, Fix A).
    cpc_gpd_loc : float
        Pre-computed CPC GPD location.
    cpc_gpd_scale : float
        Pre-computed CPC GPD scale.
    cpc_upper_cap : float
        Pre-computed CPC upper cap (99.9th percentile of wet values).
    cpc_p_threshold : float
        Pre-computed CPC gamma CDF value at the GPD threshold.
    p_dry_cpc : float
        Pre-computed CPC dry-day fraction (fraction of days <= WET_DAY_THRESHOLD).
        Required for Cannon 2015 §3.2 dry-day handling (Fix C).

    Returns
    -------
    numpy.ndarray
        Corrected precipitation values after quantile mapping.

    Notes
    -----
    Fixes applied (2026.04):
      * Fix C (Cannon 2015 §3.2): IMERG gamma fitted on wet-day values only;
        Cannon dry-day handling uses p_dry_imerg (per pixel) and p_dry_cpc
        (interpolated from native CPC fit) to match dry-day frequencies.
      * Fix B: GPD substitution triggered on IMERG-side 80th percentile of
        wet values, with conditional probability computed in the unconditional
        CDF space for consistency between gamma body and GPD tail.
      * Fix A (Coles 2001): inherited via cpc_gpd_* which were fitted with
        floc=0 upstream in cross_validate_gpd.
    """
    original_shape = imerg_values.shape
    imerg_flat = imerg_values.flatten()

    # Check if any CPC parameter is NaN → skip this pixel
    if (np.isnan(cpc_gamma_shape) or np.isnan(cpc_gamma_scale) or
            np.isnan(cpc_gpd_threshold) or np.isnan(p_dry_cpc)):
        return np.full(original_shape, np.nan)

    # Remove NaN from IMERG
    valid_mask = ~np.isnan(imerg_flat)
    imerg_valid = imerg_flat[valid_mask]

    if imerg_valid.size == 0:
        return np.full(original_shape, np.nan)

    # Check for constant IMERG values
    if np.all(imerg_valid == imerg_valid[0]) and imerg_valid[0] == 0:
        return np.zeros(original_shape)

    # ---- Fix C: wet-day sample and dry-day fraction ----
    wd = WET_DAY_THRESHOLD
    is_wet_imerg = imerg_valid > wd
    p_dry_imerg = 1.0 - is_wet_imerg.mean()

    imerg_wet = imerg_valid[is_wet_imerg]
    if len(imerg_wet) < 5:
        # Too few wet days - output all zeros
        corrected_flat = np.full_like(imerg_flat, np.nan)
        corrected_flat[valid_mask] = 0.0
        return corrected_flat.reshape(original_shape)

    # Fit IMERG gamma on wet-day sample only
    shape1, _loc1, scale1 = fit_gamma_distribution(imerg_wet)
    if shape1 <= 0 or scale1 <= 0:
        return np.full(original_shape, np.nan)

    # Initialise output (dry days and killed drizzle stay at 0)
    out = np.zeros_like(imerg_valid)

    # Wet-CDF for imerg_wet -> unconditional CDF in the mixed dry/wet model
    y_wet = gamma.cdf(imerg_wet, shape1, loc=0, scale=scale1)
    y_uncond = p_dry_imerg + (1.0 - p_dry_imerg) * y_wet

    # Cannon dry-day handling: kill drizzle below CPC dry-day frequency
    kill = y_uncond < p_dry_cpc
    keep = ~kill

    y_cond = np.zeros_like(y_uncond)
    if (1.0 - p_dry_cpc) > 1e-10:
        y_cond[keep] = (y_uncond[keep] - p_dry_cpc) / (1.0 - p_dry_cpc)
    y_cond = np.clip(y_cond, 1e-10, 1 - 1e-10)

    # Map through precomputed CPC gamma (wet-only fit)
    cpc_wet_q = gamma.ppf(y_cond, cpc_gamma_shape, loc=0, scale=cpc_gamma_scale)
    # Enforce wet-day threshold: gamma(loc=0) can map low quantiles below
    # WET_DAY_THRESHOLD even though it was fitted on wet-only data (>= 1mm).
    # Values the Cannon construction designated as wet must stay >= 1mm.
    cpc_wet_q[keep] = np.where(
        cpc_wet_q[keep] < WET_DAY_THRESHOLD, WET_DAY_THRESHOLD, cpc_wet_q[keep]
    )
    out_wet = np.where(kill, 0.0, cpc_wet_q)

    # ---- Fix B: GPD substitution on IMERG-side threshold ----
    imerg_thr_wet = np.percentile(imerg_wet, GPD_THRESHOLD_PERCENTILE)
    p_thr_wet = gamma.cdf(imerg_thr_wet, shape1, loc=0, scale=scale1)
    y_thr_uncond = p_dry_imerg + (1.0 - p_dry_imerg) * p_thr_wet

    extreme_mask_wet = imerg_wet > imerg_thr_wet
    if np.any(extreme_mask_wet):
        denom = 1.0 - y_thr_uncond
        if denom > 1e-10:
            y_ext = y_uncond[extreme_mask_wet]
            p_cond = (y_ext - y_thr_uncond) / denom
            p_cond = np.clip(p_cond, 1e-10, 1 - 1e-10)
            gpd_excess = genpareto.ppf(
                p_cond, cpc_gpd_shape, loc=cpc_gpd_loc, scale=cpc_gpd_scale
            )
            out_wet[extreme_mask_wet] = gpd_excess + cpc_gpd_threshold

    # Place wet-day outputs back
    out[is_wet_imerg] = out_wet

    # Upper cap
    if not np.isnan(cpc_upper_cap):
        out = np.minimum(out, cpc_upper_cap)

    # Non-negative
    out = np.maximum(out, 0)

    # Rebuild full array
    corrected_flat = np.full_like(imerg_flat, np.nan)
    corrected_flat[valid_mask] = out

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

    Notes
    -----
    Fixes applied (2026.04):
      * Fix C (Cannon 2015 §3.2): gamma distributions are fitted on wet-day
        values only (> WET_DAY_THRESHOLD). The dry-day fraction is handled
        explicitly via the unconditional CDF construction, and sub-threshold
        IMERG values are mapped either to zero (when the unconditional
        probability is below the CPC dry-day fraction) or through the
        conditional wet CDF of CPC. This prevents drizzle days from
        contaminating the gamma body and matches the CPC dry-day frequency.
      * Fix B: GPD substitution is triggered on IMERG's own 80th percentile
        of wet values, not on the CPC threshold, and the conditional
        probability is computed consistently in unconditional CDF space.
      * Fix A (Coles 2001): GPD fitting uses floc=0 internally.
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

    # Edge case: no data (ocean/masked pixels - no warning expected)
    if imerg_valid.size == 0 or cpc_valid.size == 0:
        return np.full(original_shape, np.nan)

    # Edge case: constant values
    if np.all(imerg_valid == imerg_valid[0]):
        if np.all(imerg_valid == 0):
            return np.zeros(original_shape)

    # ---- Fix C: split into wet-day samples ----
    wd = WET_DAY_THRESHOLD
    is_wet_imerg = imerg_valid > wd
    is_wet_cpc = cpc_valid > wd
    p_dry_imerg = 1.0 - is_wet_imerg.mean()
    p_dry_cpc = 1.0 - is_wet_cpc.mean()

    imerg_wet = imerg_valid[is_wet_imerg]
    cpc_wet = cpc_valid[is_wet_cpc]

    if len(imerg_wet) < 5 or len(cpc_wet) < 5:
        # Too few wet days for a reliable mapping - return all zeros (dry pixel).
        corrected_values_flat = np.full_like(imerg_flat, np.nan)
        corrected_values_flat[valid_mask] = 0.0
        return corrected_values_flat.reshape(original_shape)

    # Fit gammas on wet-day samples only
    shape1, _loc1, scale1 = fit_gamma_distribution(imerg_wet)
    shape2, _loc2, scale2 = fit_gamma_distribution(cpc_wet)
    if shape1 <= 0 or scale1 <= 0 or shape2 <= 0 or scale2 <= 0:
        return np.full(original_shape, np.nan)

    # Initialise output to zero (dry days, and sub-drizzle days become 0)
    out = np.zeros_like(imerg_valid)

    # ---- Cannon 2015 §3.2 dry-day handling for WET IMERG values ----
    # Wet-CDF for imerg_wet -> unconditional CDF in the mixed dry/wet model
    y_wet = gamma.cdf(imerg_wet, shape1, loc=0, scale=scale1)
    y_uncond = p_dry_imerg + (1.0 - p_dry_imerg) * y_wet

    # Values whose unconditional probability falls below CPC's dry-day
    # frequency must come out as zero (kills over-detected drizzle).
    kill = y_uncond < p_dry_cpc
    keep = ~kill

    # Conditional probability in the CPC wet distribution
    y_cond = np.zeros_like(y_uncond)
    if (1.0 - p_dry_cpc) > 1e-10:
        y_cond[keep] = (y_uncond[keep] - p_dry_cpc) / (1.0 - p_dry_cpc)
    y_cond = np.clip(y_cond, 1e-10, 1 - 1e-10)

    out_wet = gamma.ppf(y_cond, shape2, loc=0, scale=scale2)
    # Enforce wet-day threshold: gamma(loc=0) can map low quantiles below
    # WET_DAY_THRESHOLD even though it was fitted on wet-only data (>= 1mm).
    # Values the Cannon construction designated as wet must stay >= 1mm.
    out_wet[keep] = np.where(
        out_wet[keep] < WET_DAY_THRESHOLD, WET_DAY_THRESHOLD, out_wet[keep]
    )
    out_wet = np.where(kill, 0.0, out_wet)

    # ---- Fix B: GPD substitution on IMERG-side threshold ----
    # IMERG's own 80th percentile of wet values (not CPC's)
    imerg_thr_wet = np.percentile(imerg_wet, GPD_THRESHOLD_PERCENTILE)
    p_thr_wet = gamma.cdf(imerg_thr_wet, shape1, loc=0, scale=scale1)
    y_thr_uncond = p_dry_imerg + (1.0 - p_dry_imerg) * p_thr_wet

    extreme_mask_wet = imerg_wet > imerg_thr_wet
    if np.any(extreme_mask_wet) and len(cpc_wet) >= 10:
        cpc_thr = np.percentile(cpc_wet, GPD_THRESHOLD_PERCENTILE)
        cpc_gpd_params = cross_validate_gpd(cpc_wet, cpc_thr)

        denom = 1.0 - y_thr_uncond
        if denom > 1e-10 and not np.isnan(cpc_thr) and cpc_thr > 0:
            y_ext = y_uncond[extreme_mask_wet]
            p_cond = (y_ext - y_thr_uncond) / denom
            p_cond = np.clip(p_cond, 1e-10, 1 - 1e-10)
            gpd_excess = genpareto.ppf(p_cond, *cpc_gpd_params)
            out_wet[extreme_mask_wet] = gpd_excess + cpc_thr

    # Place wet-day outputs back into the full output array
    out[is_wet_imerg] = out_wet

    # ---- Upper cap at 99.9th percentile of CPC wet values ----
    dynamic_cap = np.percentile(cpc_wet, UPPER_CAP_THRESHOLD_PERCENTILE)
    if not np.isnan(dynamic_cap):
        out = np.minimum(out, dynamic_cap)

    # Ensure non-negative
    out = np.maximum(out, 0)

    # Rebuild full array
    corrected_values_flat = np.full_like(imerg_flat, np.nan)
    corrected_values_flat[valid_mask] = out
    corrected_values = corrected_values_flat.reshape(original_shape)
    return corrected_values
