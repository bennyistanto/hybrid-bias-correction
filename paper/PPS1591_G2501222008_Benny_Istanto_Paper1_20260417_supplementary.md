# Supplementary Material: A Scalable Framework for Enhancing Satellite-Derived Daily Precipitation - Adjusting Values, Aligning Distributions, and Preserving Extremes

**Authors:** Benny Istanto, Rizaldi Boer, I Putu Santikayasa

---

# S1. Data Preprocessing Details

To ensure comparability between the satellite-derived and ground-based datasets, several preprocessing steps were undertaken.

- **Spatial harmonization.** The CPC-UNI dataset is available at 0.5° spatial resolution, five times coarser than the 0.1° IMERG grid. To enable pixel-level correction, CPC-UNI is regridded to the IMERG grid using nearest-neighbor interpolation, which preserves the discrete gauge-analyzed values without artificial smoothing. This regridded product is used for daily time-series pairing during the correction loop. To preserve the statistical independence of the gauge-based reference during distribution fitting, the framework retains the CPC-UNI data at its native 0.5° resolution alongside the regridded product. Following the Bias Correction Spatial Disaggregation (BCSD) principle, distribution parameters and climatological means are estimated at the native 0.5° CPC-UNI resolution, then bilinearly interpolated to the 0.1° IMERG grid before quantile mapping is applied. This produces spatially smooth correction fields while preserving the statistical integrity of the gauge-based reference.

- **Temporal alignment.** Daily time steps were synchronized across all datasets, and a strictly monotonic, duplicate-free time axis was enforced. Where duplicate time stamps existed due to data processing artifacts, only the first occurrence was retained.

- **Land-sea masking.** A binary land-sea mask derived from official land boundary datasets was applied to both IMERG and CPC-UNI at data loading, setting all ocean grid cells to missing values. This prevents ocean zeros from contaminating statistical calculations (e.g., mean bias, percentile estimation) that rely on the assumption of land-based precipitation sampling.

- **Wet-day threshold.** A wet-day threshold of 1.0 mm/day is applied throughout the framework for categorical metric computation and precipitation day classification, following World Meteorological Organization (WMO) recommendations for operational precipitation monitoring in tropical regions where trace precipitation is frequent.

- **Latitude orientation.** The CPC-UNI dataset uses descending latitude ordering. To ensure consistent array operations, latitude coordinates are checked and reindexed to ascending order where necessary prior to any spatial alignment or computation.

# S2. Evaluation Metric Formulas

This section provides the mathematical definitions for the 31 verification metrics used in the evaluation framework. All metrics are computed at the pixel level for each dekadal period. A wet-day threshold of 1.0 mm/day is applied for all categorical metrics.

## S2.1 Continuous Metrics

The Relative Bias measures the proportional difference between test and reference totals:

$$\text{RB} = \frac{\sum_{i=1}^{n} P_{test,i} - \sum_{i=1}^{n} P_{ref,i}}{\sum_{i=1}^{n} P_{ref,i}}$$

The Pearson Correlation Coefficient quantifies the linear relationship:

$$r = \frac{\sum_{i=1}^{n}(P_{ref,i} - \bar{P}_{ref})(P_{test,i} - \bar{P}_{test})}{\sqrt{\sum_{i=1}^{n}(P_{ref,i} - \bar{P}_{ref})^2 \cdot \sum_{i=1}^{n}(P_{test,i} - \bar{P}_{test})^2}}$$

The Root Mean Square Error and Mean Absolute Error quantify error magnitude:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (P_{test,i} - P_{ref,i})^2}$$

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |P_{test,i} - P_{ref,i}|$$

The Nash-Sutcliffe Efficiency provides a normalized measure of prediction skill relative to the reference mean:

$$\text{NSE} = 1 - \frac{\sum_{i=1}^{n}(P_{ref,i} - P_{test,i})^2}{\sum_{i=1}^{n}(P_{ref,i} - \bar{P}_{ref})^2}$$

NSE values range from $-\infty$ to 1, where values above 0.5 indicate satisfactory performance and values below zero indicate that the reference mean is a better predictor than the corrected product.

The Standard Deviation is computed independently for both reference and test datasets using the sample estimator (with Bessel's correction, $\text{ddof} = 1$):

$$\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (P_i - \bar{P})^2}$$

The Standard Deviation Ratio (SDR) quantifies the relative variability of the corrected product compared to the reference:

$$\text{SDR} = \frac{\sigma_{test}}{\sigma_{ref}}$$

SDR values greater than 1 indicate over-dispersion, while values less than 1 indicate under-dispersion. A perfect correction preserves the reference variability ($\text{SDR} = 1$).

## S2.2 Categorical Metrics

Event detection is assessed using the probability of detection, false alarm ratio, and critical success index, defined in terms of contingency table entries (hits $H$, misses $M$, and false alarms $F$):

$$\text{POD} = \frac{H}{H + M}, \quad \text{FAR} = \frac{F}{H + F}, \quad \text{CSI} = \frac{H}{H + M + F}$$

The Equitable Threat Score (ETS) adjusts CSI by removing hits attributable to random chance:

$$\text{ETS} = \frac{H - H_{random}}{H + M + F - H_{random}}, \quad H_{random} = \frac{(H + M)(H + F)}{N}$$

where $N$ is the total number of observations. ETS = 0 indicates no skill beyond random forecasts; ETS = 1 indicates perfect detection.

## S2.3 Temporal Metrics

The Frequency of Precipitation Days (FPD) measures the percentage of days exceeding the wet-day threshold:

$$\text{FPD} = \frac{N_{wet}}{N_{valid}} \times 100$$

where $N_{wet}$ is the number of days with precipitation $\geq$ 1.0 mm and $N_{valid}$ is the total number of valid (non-missing) days.

The Mean Wet-Day Precipitation (MDWP) captures the average intensity on days that exceed the wet-day threshold:

$$\text{MDWP} = \frac{1}{N_{wet}} \sum_{i=1}^{N_{wet}} P_i \quad \text{for } P_i \geq 1.0 \text{ mm/day}$$

The Maximum Dry Spell Length (DSL) is defined as the longest consecutive run of days below the wet-day threshold:

$$\text{DSL} = \max_{j} \{ L_j \}$$

where $L_j$ is the length of the $j$-th consecutive dry spell.

## S2.4 Distributional Metrics

Six percentiles ($Q_{25}$, $Q_{50}$, $Q_{75}$, $Q_{90}$, $Q_{95}$, $Q_{99}$) are computed for both reference and test datasets. The two-sample Kolmogorov-Smirnov test evaluates distributional agreement:

$$D_{KS} = \sup_{x} |F_{test}(x) - F_{ref}(x)|$$

Metrics are computed in two modes: (1) a timeseries mode producing one metric grid per year, and (2) a single-dekad mode pooling all years into one aggregated assessment.

## S2.5 CQI Sub-Score Formulas

Each percentile score is computed as:

$$S_{Q_p} = 1 - \min\left(\frac{|Q_{p,test} - Q_{p,ref}|}{Q_{p,ref} + 0.1}, \; 1\right)$$

The variability score:

$$S_{var} = 1 - \min(|1 - \text{SDR}|, \; 1)$$

The event detection composite:

$$S_{event} = 0.6 \cdot \text{POD} + 0.4 \cdot (1 - \text{FAR})$$

The spell preservation score:

$$S_{spell} = 1 - \min\left(\frac{|\text{DSL}_{test} - \text{DSL}_{ref}|}{\text{DSL}_{ref} + \epsilon}, \; 1\right)$$

where $\epsilon = 10^{-6}$ prevents division by zero.

**Table S1.** CQI categorical classification

     Class      CQI Range    Interpretation
  ----------- -------------- ------------------------------------------------------------------
   Excellent    $\geq 0.8$   Correction meets high-performance standards across all metrics
     Good      $[0.6, 0.8)$  Well-performing with minor deficiencies in some areas
     Fair      $[0.4, 0.6)$  Basic improvements achieved but moderate performance gaps remain
     Poor        $< 0.4$     Substantial deficiencies and limited reliability

# S3. Confidence Assessment

The confidence score provides an additional layer of information indicating the reliability of the computed quality metrics. Because the statistical stability of pixel-level metrics depends on sample size, metric consistency, and distributional agreement, the confidence computation is adapted to the evaluation mode.

For the aggregated (single-dekad) evaluation, where sample sizes are large by design ($\sim$230 daily values pooled across all years):

$$C = 0.60 \cdot S_{consistency} + 0.40 \cdot S_{dist}$$

For the timeseries evaluation, where sample sizes vary per year ($\sim$10 daily values per dekad):

$$C = 0.40 \cdot S_{sample} + 0.30 \cdot S_{dist} + 0.30 \cdot S_{consistency}$$

where $S_{sample}$ is the fraction of valid (non-missing) time steps, $S_{consistency}$ measures agreement among normalized basic metrics computed as:

$$S_{consistency} = 1 - \text{std}\left(\tilde{S}_{RB}, \; \tilde{S}_{NSE}, \; \text{POD}, \; 1 - \text{FAR}\right)$$

where $\tilde{S}$ denotes the normalized score (mapped to $[0, 1]$ where 1 is optimal), and $S_{dist}$ is the Kolmogorov-Smirnov test $p$-value.

Grid cells or time periods with low confidence scores are flagged, highlighting regions where the evaluation may be less reliable due to data limitations or inconsistent correction behavior.

# S4. Conditional Bias Decomposition

The domain-average relative bias reported in the main text conceals an important spatial-conditional structure. A per-pixel decomposition of the relative bias was performed by partitioning each pixel's time series into three regions defined on the CPC-UNI reference distribution: R1 (body, reference values $\leq Q_{80}$), R2 (moderate tail, $Q_{80} < \text{ref} \leq Q_{99.9}$), and R3 (extreme, ref $> Q_{99.9}$). The sum-weighted contribution of each region to the overall domain bias is reported as a percentage-point (pp) share of the reference total.

**Table S2.** Per-pixel R1/R2/R3 decomposition of sum-weighted relative bias against CPC-UNI for January dekad 1, land pixels only. Contributions are expressed in percentage points of the reference sum and are additive to the overall RB.

  Field                  Overall RB (%)   R1 contrib (pp)             R2 contrib (pp)            R3 contrib (pp)
  -------------------- ---------------- ----------------- --------------------------- --------------------------
  Raw IMERG-L                    +15.52            +38.91   $-$20.71   $-$2.68
  LSEQM (this study)              +7.90            +31.08   $-$20.59   $-$2.60

The decomposition reveals a persistent signature conserved across correction stages: IMERG is systematically *too wet* on days that CPC-UNI classifies as light-rain days (R1, $+27$ to $+39$ pp) and simultaneously *too dry* on days that CPC-UNI classifies as moderate-heavy or extreme (R2 $-21$ pp, R3 $-2.6$ pp). The two sign-opposite errors partially cancel in the marginal summary, but at the pixel-day level they reflect a mismatch in *which days are heavy where* rather than a mismatch in the marginal distribution of daily intensities per pixel. This is a conditional, not a marginal, bias.

A conditional mismatch of this form cannot be removed by univariate quantile mapping by construction, because quantile mapping adjusts the marginal CDF of each pixel without modifying the temporal assignment of values to days. The LSEQM correction visibly flattens R1 (from $+38.9$ to $+31.1$ pp) and leaves R2 and R3 essentially unchanged, consistent with a marginal correction acting on an underlying conditional error.
