# Implementation

This chapter describes how you can begin the practical execution of the bias correction workflow. Unlike the theoretical guidelines, these notebooks are designed for hands-on application. Before you begin, ensure you have followed the [Setup](setup.md) instructions to configure your environment correctly—whether you are using Google Colab or running locally.

## Overview

The project includes four primary notebooks that guide you through each stage of the bias correction process:

- **Notebook 1: Data Download and Area of Interest Preparation**  
  This notebook guides you through downloading the necessary satellite and observation datasets, and preparing your area of interest (AOI). You will verify data integrity and ensure that the required data are correctly georeferenced.

- **Notebook 2: Executing the Bias Correction Workflow**  
  In this notebook, you will run the full bias correction process using the hybrid method. This includes:
  - Adjusting values with Linear Scaling (LS)  
  - Aligning distributions with Empirical Quantile Mapping (EQM) augmented by gamma fitting and tail adjustment via a Generalized Pareto Distribution (GPD)  
  - Incorporating the Deep Learning (DL) enhancement to refine the corrected output, especially for extreme values.

- **Notebook 3: Performance Evaluation**  
  This notebook provides tools and metrics to measure the performance of the bias correction process. You will compute and analyze key statistical metrics (such as RMSE, bias, and correlation) to evaluate the improvement in the corrected precipitation compared to observed data.

- **Notebook 4: Quality Assessment and Visualization**  
  The final notebook is dedicated to conducting a thorough quality assessment. It contains various visualization tools (e.g., plots, maps) that help you assess the spatial and temporal consistency of the bias-corrected data.

## Getting Started

### Checklist

Before running the notebooks, confirm that you have completed the following from the [Setup](setup.md) chapter:

- Mounted your Google Drive (if using Colab) or set up your local environment.
- Cloned the repository from GitHub: [https://github.com/bennyistanto/hybrid-bias-correction](https://github.com/bennyistanto/hybrid-bias-correction)
- Installed all required packages as specified in `requirements.txt` (applied only for local implementation)

### How to Run the Notebooks

1. **Open Notebook 1: Data Download and AOI Preparation**  
   Navigate to the `notebooks/` folder in your Colab or local Jupyter Notebook environment and open the first notebook. Follow the step-by-step instructions and checklist within the notebook to download data and set your area of interest.

2. **Open Notebook 2: Bias Correction Execution**  
   With your data prepared, open the second notebook which implements the bias correction workflow. Follow the guided steps to execute the LS, EQM (with gamma and GPD tail adjustment), and DL enhancement processes.

3. **Open Notebook 3: Performance Evaluation**  
   After obtaining the bias-corrected data, use the third notebook to evaluate performance metrics. This notebook offers statistical analysis tools to compare corrected data against observed precipitation.

4. **Open Notebook 4: Quality Assessment and Visualization**  
   Finally, open the fourth notebook to visualize the bias-corrected results. The visualizations help assess the spatial and temporal quality of the adjustments, ensuring that extreme events and typical values are accurately represented.

By following these sequential steps, you can efficiently run through the bias correction workflow and gain insights into both the performance and quality of the corrected precipitation estimates.

Happy Processing!
