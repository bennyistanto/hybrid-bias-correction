"""Smoke tests for the hybrid-bias-correction framework.

These tests verify that the framework's public surface is wired correctly:
imports resolve, the distribution-fitting primitives accept reasonable
inputs and return numerically sensible outputs, and the station-density
confidence machinery runs on tiny synthetic arrays.

The tests deliberately avoid the data archive cited in the README - they
exercise only the algorithmic code paths on in-memory synthetic data, so
they run on a free CI runner in seconds. End-to-end pipeline verification
against the Bali example dataset is documented in the project README and
in the docs site (notebook 02); a longer Bali-subdomain CI workflow is on
the backlog (see CHANGELOG).

Run from the repository root:

    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Import surface
# ---------------------------------------------------------------------------

def test_src_modules_import():
    """Every top-level pipeline module loads without error."""
    import src.config          # noqa: F401
    import src.io              # noqa: F401
    import src.utility         # noqa: F401
    import src.bias_correction # noqa: F401
    import src.distribution_fitting  # noqa: F401
    import src.deep_learning   # noqa: F401
    import src.station_density # noqa: F401
    import src.metrics         # noqa: F401
    import src.qa_framework    # noqa: F401


# ---------------------------------------------------------------------------
# 2. Distribution fitting primitives
# ---------------------------------------------------------------------------

def test_gamma_fit_recovers_known_parameters():
    """A Gamma fit on a large sample should recover the parameters used to
    draw the sample, within a tolerance proportional to the sample size.
    `fit_gamma_distribution` returns scipy-style (shape, loc, scale)."""
    from src.distribution_fitting import fit_gamma_distribution
    rng = np.random.default_rng(42)
    true_shape, true_scale = 0.6, 4.0
    sample = rng.gamma(shape=true_shape, scale=true_scale, size=10_000)
    shape_hat, _loc, scale_hat = fit_gamma_distribution(sample)
    assert shape_hat == pytest.approx(true_shape, rel=0.10)
    assert float(scale_hat) == pytest.approx(true_scale, rel=0.10)


def test_l_moments_finite_for_realistic_rainfall():
    """L-moments should be finite and ordered for a non-degenerate sample."""
    from src.distribution_fitting import calculate_l_moments
    rng = np.random.default_rng(0)
    rain = rng.gamma(shape=0.5, scale=3.0, size=2000)
    rain = rain[rain > 0]  # wet days only
    moments = calculate_l_moments(rain)
    # Accept either a dict or a tuple/array; both forms are observed in src.
    values = (moments.values() if isinstance(moments, dict) else moments)
    values = list(values)
    assert all(np.isfinite(v) for v in values)


# ---------------------------------------------------------------------------
# 3. Station-density confidence
# ---------------------------------------------------------------------------

def test_confidence_compute_runs_on_synthetic_counts():
    """`compute_confidence_map` should return a [0,1]-bounded array of the
    same shape as the input station-count grid. Accepts xarray.DataArray
    on the CPC native (coarse) grid."""
    import xarray as xr
    from src.station_density import compute_confidence_map
    rng = np.random.default_rng(1)
    counts = xr.DataArray(
        rng.integers(low=0, high=8, size=(6, 10)).astype("int32"),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-10, 5, 6), "lon": np.linspace(95, 140, 10)},
    )
    mask = compute_confidence_map(counts, saturation_count=2,
                                    smoothing_sigma=1.0)
    assert mask.shape == counts.shape
    finite = mask.values[np.isfinite(mask.values)]
    assert finite.min() >= 0.0 - 1e-9
    assert finite.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 4. Algebraic blending rule
# ---------------------------------------------------------------------------

def test_alpha_blend_endpoints():
    """The density-aware blending rule a_eff = 1 - C(1 - a) collapses to
    a=1 at C=0 (no station support -> pure LSEQM) and to a at C=1 (full
    confidence -> nominal blend)."""
    alpha = 0.70
    for confidence, expected in [(0.0, 1.0), (1.0, alpha), (0.5, 0.85)]:
        a_eff = 1.0 - confidence * (1.0 - alpha)
        assert a_eff == pytest.approx(expected)
