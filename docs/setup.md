# Setup

This chapter describes how to set up the Hybrid Bias Correction project. The project is hosted on GitHub at [https://github.com/bennyistanto/hybrid-bias-correction](https://github.com/bennyistanto/hybrid-bias-correction). We have two options for running the project:

1. **Using Google Colab** (recommended) – This minimizes the need to install many packages locally and offers GPU support for training the deep learning model.
2. **Running Locally** – This option is for users who prefer to work on their own machines.

---

## 1. Setting Up in Google Colab

### Option A: Clone the Repository into Our Google Drive

Cloning the repository into our Google Drive ensures that our work is persistent even if the Colab runtime is reset.

1. **Access Google Colab**

   Access Google Colab via browser [https://colab.research.google.com/](https://colab.research.google.com/) then click New Notebook.

2. **Mount Our Google Drive:**

   Run the following code in a new cell to mount our drive:

   ```python
   from google.colab import drive
   import os
   
   # Check if the drive is mounted
   if os.path.exists('/content/drive'):
      # Try to unmount
      try:
         drive.flush_and_unmount()
         print("Successfully unmounted")
      except:
         print("Unmount failed, the drive might not be mounted or busy")
   
   # Mount the drive
   drive.mount('/content/drive')
   ```

3. **Clone the Repository:**

   Choose a folder in our Google Drive (for example, `/content/drive/MyDrive/hybrid-bias-correction`) and run:

   ```bash
   !git clone https://github.com/bennyistanto/hybrid-bias-correction.git "/content/drive/MyDrive/hybrid-bias-correction"
   ```

4. **Navigate to the Repository Folder:**

   Change our working directory to the cloned repository:

   ```python
   import os
   os.chdir("/content/drive/MyDrive/hybrid-bias-correction")
   !ls
   ```

5. **Run the Notebooks:**

   Open an example notebook from the `notebooks/` folder and execute the cells. The notebooks are designed to install packages on the fly if needed.

---

### Option B: Clone the Repository Directly in Colab

If we prefer not to use Google Drive, we can clone the repository directly into the Colab environment.

> [!CAUTION]
>
> Note that the environment is temporary and will be reset when the runtime is restarted.

1. **Access Google Colab**

   Access Google Colab via browser [https://colab.research.google.com/](https://colab.research.google.com/) then click New Notebook.

2. **Clone the Repository:**

   Run the following code in a new cell to clone the repository.

   ```bash
   !git clone https://github.com/bennyistanto/hybrid-bias-correction.git
   ```

3. **Navigate to the Repository Folder:**

   ```python
   import os
   os.chdir("hybrid-bias-correction")
   !ls
   ```

4. **Run the Notebooks:**

   Open and run the notebooks from the `notebooks/` folder.

---

## 2. Setting Up Locally

If we prefer to run the project on our local machine, follow these steps:

1. **Clone the Repository:**

   Open a terminal and run:

   ```bash
   git clone https://github.com/bennyistanto/hybrid-bias-correction.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd hybrid-bias-correction
   ```

3. **Install Required Packages:**

   Ensure we have Python 3 installed. Then, install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   This command installs all dependencies (e.g., xarray, numpy, netCDF4, TensorFlow, etc.) needed to run the project.

4. **Run the Notebooks Locally:**

   If we want to run the notebooks locally, we can launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Navigate to the `notebooks/` folder and open an example notebook to run the bias correction workflow.

---

## Additional Notes

- **Google Colab Advantages:**  
  Running on Colab minimizes local installation hassles and provides free GPU support for the deep learning model training.

- **Local Execution Considerations:**  
  It is recommended to use a virtual environment (e.g., via `venv` or `conda`) to manage dependencies for the project.

- **Keeping the Repository Updated:**  
  If we need to update our local or Colab clone, navigate to the repository directory and run:

  ```bash
  git pull
  ```

By following these steps, we can set up and run the Hybrid Bias Correction project either on Google Colab or on our local machine.
