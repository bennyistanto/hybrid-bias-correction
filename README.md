# Hybrid Bias Correction

Hybrid Bias Correction (LSEQM+DL) is a Python-based workflow for correcting biases in daily precipitation data. It combines traditional methods—Linear Scaling (LS) and Empirical Quantile Mapping (EQM)—with an optional Deep Learning (DL) component to improve the correction, particularly for extreme events.

This repository is designed to be run on [Google Colab](https://colab.research.google.com/). The source code is organized in the `src` folder, and example notebooks are provided in the `notebooks` folder.

![LSEQM+DL](docs/images/lseqmdl.png "LSEQM+DL")

## Repository Structure

```
hybrid-bias-correction/
├── src/
│   ├── __init__.py           # Empty file to mark src as a Python package
│   ├── config.py             # Default configuration (paths, parameters, and filename template)
│   ├── utility.py            # Utility functions for data preparation and I/O management
│   ├── distribution_fitting.py   # Statistical functions for L-moments, gamma and GPD fitting, quantile mapping
│   ├── io.py                 # Functions for file I/O, saving NetCDF files, and data aggregation
│   ├── deep_learning.py      # Functions for training and applying the deep learning model
│   └── bias_correction.py    # High-level bias correction workflow (LS + EQM [+ DL])
├── notebooks/
│   └── colab_notebook.ipynb  # Example notebook for running the workflow on Colab
├── README.md                 # This file
└── requirements.txt          # List of required Python packages
```

## Getting Started on Colab

To run the workflow on Colab, follow these steps:

1. **Clone the Repository to Google Drive:**

   Upload or clone the repository into your Google Drive. For example, place it under:
   ```
   /content/drive/MyDrive/exercises/hybrid-bias-correction
   ```

2. **Open the Example Notebook:**

   Open the notebook from the `notebooks` folder (e.g., `colab_notebook.ipynb`) using Colab.

3. **Mount Your Google Drive:**

   In the first cell of the notebook, mount your Google Drive so that the code can access the repository and data files:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Set Up the Python Path:**

   Add the `src` directory to the Python path so that the notebook can import the modules:

   ```python
   import sys, os
   src_path = os.path.abspath('/content/drive/MyDrive/exercises/hybrid-bias-correction/src')
   if src_path not in sys.path:
       sys.path.append(src_path)
   ```

5. **Configure File Paths (if needed):**

   You can override the default file paths by specifying them in the notebook cell. For example:

   ```python
   import os
   MAIN_DIR = '/content/drive/MyDrive/exercises/hybrid-bias-correction'
   INPUT_DIR = os.path.join(MAIN_DIR, 'data/bc/input')
   OUTPUT_DIR = os.path.join(MAIN_DIR, 'data/bc/output')
   IMERG_FILE = os.path.join(INPUT_DIR, 'imergl/idn_imergl.nc4')
   CPC_FILE = os.path.join(INPUT_DIR, 'cpcuni/idn_cpcuni.nc4')
   MASK_FILE = os.path.join(MAIN_DIR, 'data/subset/iso3/idn_subset.nc')
   ```

6. **Execute the Workflow:**

   Finally, run the main execution function (for example, using the high-level function from `bias_correction.py`):

   ```python
   from bias_correction import lseqmdf
   import xarray as xr
   
   # Load datasets
   imerg_ds = xr.open_dataset(IMERG_FILE, decode_times=True)
   cpc_ds = xr.open_dataset(CPC_FILE, decode_times=True)
   
   # Specify the month and dekad (as two-digit strings)
   month = 5
   dekad_start_day, dekad_end_day = 1, 10  # Example for first dekad
   month_str = "05"
   dekad_str = "01"
   
   # Optionally, define output directories (overriding defaults)
   ls_corrected_precip_path = os.path.join(OUTPUT_DIR, 'corrected_ls')
   lseqm_corrected_precip_path = os.path.join(OUTPUT_DIR, 'corrected_lseqm')
   lseqmdl_corrected_precip_path = os.path.join(OUTPUT_DIR, 'corrected_lseqmdl')
   
   # Run the bias correction workflow
   corrected_data = lseqmdf(
       imerg_ds, cpc_ds, month, dekad_start_day, dekad_end_day,
       month_str=month_str, dekad_str=dekad_str,
       ls_corrected_precip_path=ls_corrected_precip_path,
       lseqm_corrected_precip_path=lseqm_corrected_precip_path,
       lseqmdl_corrected_precip_path=lseqmdl_corrected_precip_path
   )
   ```

## Installation

Install the required Python packages using the provided `requirements.txt` file. In Colab, you can run:

```bash
!pip install -r /content/drive/MyDrive/exercises/hybrid-bias-correction/requirements.txt
```

## Configuration

All default settings (paths, filename template, statistical parameters, and DL hyperparameters) are located in `src/config.py`. You can modify these defaults or override them in your notebook.

## Contributing

Contributions and suggestions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the Mozilla Public License 2.0.
See LICENSE for the full license text or visit https://www.mozilla.org/en-US/MPL/2.0/.

## Authors

[Benny Istanto](https://benny.istan.to)<br>

* GOST/DECSC/DEC Data Group, The World Bank, United States ([bistanto@worldbank.org](mailto:bistanto@worldbank.org))
* Applied Climatology Study Program, Bogor Agricultural University, Indonesia ([bistanto@ipb.ac.id](mailto:bistanto@ipb.ac.id))

With supervision from [Prof. Rizaldi Boer](https://scholar.google.co.id/citations?user=jTPXEp8AAAAJ&hl=id) and [Dr. I Putu Santikayasa](https://scholar.google.com/citations?user=DcQ58z8AAAAJ&hl=en)