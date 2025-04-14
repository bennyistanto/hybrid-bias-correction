# Repository Structure

The source code for the Hybrid Bias Correction project is hosted on GitHub at [https://github.com/bennyistanto/hybrid-bias-correction](https://github.com/bennyistanto/hybrid-bias-correction). The repository is organized into several key folders:

- **data/**: (Optional) Stores the input and output data or references to where data can be found.
- **docs/**: Contains the documentation for the project.
- **notebooks/**: Includes example notebooks demonstrating how to run the workflow in a practical setting.
- **src/**: Contains the core Python modules that implement the bias correction workflow.
- **README**: Provides an overview of the project and basic instructions.
- **requirements**: Lists the Python packages required for the project.

Below is a snapshot of the repository structure:

```graphql
hybrid-bias-correction/
├── data/                        # Optionally place for store the data
├── docs/                        # Documentation for the project
├── notebooks/                   # Example notebook for running the workflow on Colab
│   └── 01_preparing_area_of_interest.ipynb 
│   └── 02_lseqmdl_bias_correction.ipynb
│   └── 03_performance_quality_assessment.ipynb
│   └── 04_vizualisation.ipynb
├── src/
│   ├── __init__.py              # Marks src as a Python package
│   ├── config.py                # Default configuration (paths, parameters, and filename template)
│   ├── utility.py               # Utility functions for data preparation and I/O management
│   ├── distribution_fitting.py  # Functions for calculating L-moments, gamma and GPD fitting, and quantile mapping
│   ├── io.py                    # Functions for file I/O, saving NetCDF files, and data aggregation
│   ├── deep_learning.py         # Functions for training and applying the deep learning model
│   └── bias_correction.py       # High-level bias correction workflow (LS + EQM [+ DL])
├── README.md                    # Project overview and instructions
└── requirements.txt             # List of required Python packages
```
