"""
Module: distribution_fitting.py

This module contains functions for statistical distribution fitting and quantile mapping
used in the bias correction workflow. It includes functions for:
  - Calculating L-moments and L-moment ratios.
  - Fitting a gamma distribution using L-moments.
  - Fitting a Generalized Pareto Distribution (GPD) to data above a threshold.
  - Performing K-Fold cross-validation to obtain stable GPD parameters.
  - Applying gamma distribution-based quantile mapping with tail adjustment.

Dependencies:
  - Imports default parameters from the config module.
  - Uses scipy.stats and sklearn.model_selection for statistical calculations.

**Author**:
Benny Istanto
- GOST/DECSC/DEC Data Group, The World Bank, United States. Email: bistanto@worldbank.org
- Applied Climatology Study Program, Bogor Agricultural University, Indonesia. Email: bennyistanto@ipb.ac.id

with supervision from Prof. Rizaldi Boer and Dr. I Putu Santikayasa

Update: 2025
"""
# Import the library
import numpy as np
import logging
from scipy.stats import gamma, genpareto
from sklearn.model_selection import KFold
from .config import N_SPLITS_GPD_CROSSVALIDATE, GPD_THRESHOLD_PERCENTILE, UPPER_CAP_THRESHOLD_PERCENTILE

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

# ----
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

# ----
# To address the issue of capturing extreme values in satellite data while performing EQM,
# Tail Adjustment is use to improve fit for extreme value. Tail adjustment with
# the Generalized Pareto Distribution (GPD) better captures the extreme values by specifically
# modeling the tails of the distribution, which iscrucial for accurately representing
# extreme precipitation events.

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
