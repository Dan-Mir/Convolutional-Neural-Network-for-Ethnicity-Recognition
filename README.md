# Convolutional Neural Network for Ethnicity Recognition

This repository contains an experimental Convolutional Neural Network (CNN) project for ethnicity recognition using still face pictures. The code, data samples, and training metrics are provided to reproduce, inspect or extend the experiments.

The project is an academic/research prototype. It contains the dataset snapshots used for training/validation, a single entrypoint script `main.py`, and CSV metric exports for several training runs.

## Repository structure

- `main.py` — project entrypoint. See the file for how the dataset is loaded and how training/evaluation are orchestrated.
- `data/` — sample still-picture datasets that were used for the experiments. The folder contains two main subfolders:
  - `Still_pictures_Mediterranean_ZIPfile_approx_40Mb/`
    - `Apex Stills/`
      - `Extras and HD Stills/`
    - `Prototype Stills/`
      - `Extras and HD Stills/`
  - `Still_pictures_NorthEuropean_ZIPfile_approx_50Mb/`
    - `Apex Stills/`
      - `Extras and HD Stills/`
    - `Prototype Stills/`
      - `Extras and HD Stills/`
  
  Note: the repository contains the original still images (or a representative subset) organized into these folders. The dataset appears to have two classes (Mediterranean and North European) based on the folder names.

- `metrics/` — CSV exports of training/validation metrics recorded at the end of runs. Filenames encode experimental settings (epochs and activation functions). The following CSV files are included:
  - `metrics_epoch_10_act1_gelu_act2_softmax.csv`
  - `metrics_epoch_10_act1_gelu_act2_tanh.csv`
  - `metrics_epoch_10_act1_relu_act2_sigmoid.csv`
  - `metrics_epoch_10_act1_relu_act2_softmax.csv`
  - `metrics_epoch_15_act1_gelu_act2_softmax.csv`
  - `metrics_epoch_15_act1_gelu_act2_tanh.csv`
  - `metrics_epoch_15_act1_relu_act2_sigmoid.csv`
  - `metrics_epoch_15_act1_relu_act2_softmax.csv`
  - `metrics_epoch_30_act1_gelu_act2_softmax.csv`
  - `metrics_epoch_30_act1_relu_act2_sigmoid.csv`
  - `metrics_epoch_30_act1_relu_act2_softmax.csv`
  - `metrics_epoch_40_act1_gelu_act2_softmax.csv`
  - `metrics_epoch_40_act1_relu_act2_sigmoid.csv`
  - `metrics_epoch_40_act1_relu_act2_softmax.csv`

- `images/` — (empty in the repository snapshot). This directory can be used to store visualizations such as training curves or example predictions.
- `LICENSE` — project license file.

## What this README documents

This README summarizes:

- the dataset layout and likely classes (derived from folder names),
- the available metric exports and how to interpret their filenames,
- how to run `main.py` (high-level) and where to look for results,
- suggestions for reproducing or extending the experiments.

I did not invent any code, hyperparameters, or external components that are not present in the repository. If you need the README to include sample outputs (tables/plots), provide the specific CSV rows or images and I will add them.

## Interpreting the `metrics/` files

Each CSV file contains logged metrics from a training run. Filenames follow this pattern:

`metrics_epoch_<N>_act1_<A1>_act2_<A2>.csv`

Where:
- `<N>` is the number of epochs used in the run (10, 15, 30, 40 in the included files).
- `<A1>` is the activation function used in the first part of the model (for example `relu` or `gelu`).
- `<A2>` is the activation used at the model output or second part (for example `softmax`, `tanh`, `sigmoid`).

These files can be opened in any spreadsheet program or parsed with Python/Pandas. They typically contain per-epoch metrics such as training and validation loss and accuracy. Because the repository includes multiple runs with different activations and epochs, the CSVs enable comparison across architectures and training durations.

Example quick inspection in Python:

```python
import pandas as pd
df = pd.read_csv('metrics/metrics_epoch_30_act1_relu_act2_softmax.csv')
print(df.head())
```

## Running the project

The repository provides `main.py` as the entrypoint. It is expected to run with a Python 3 environment and standard ML libraries (TensorFlow, PyTorch, scikit-learn, etc.) depending on the implementation inside `main.py`.

Because the repository does not include a `requirements.txt` or explicit environment file, create an isolated environment and install commonly used packages for CNN experiments. Example (Windows PowerShell):

```powershell
python -m venv .venv; 
.\.venv\Scripts\Activate.ps1; 
pip install --upgrade pip; 
pip install numpy pandas matplotlib pillow
# Install one of the deep learning frameworks used by your code (choose one):
# pip install tensorflow
# or
# pip install torch torchvision
```

Then run the main script:

```powershell
python main.py
```

If `main.py` depends on specific packages or command-line options, open and inspect the file to see the exact parameters. The top of `main.py` typically documents or imports the required libraries.

## Results and comparisons

The `metrics/` folder contains multiple runs that vary by epoch count and activation functions. Use the CSVs to build comparison plots. Typical comparisons include:

- Training/validation loss vs epoch
- Training/validation accuracy vs epoch
- Final epoch metrics across runs

Suggested Python snippet to compare accuracy across multiple CSVs:

```python
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

files = glob('metrics/metrics_epoch_*_act1_*_act2_*.csv')
for f in files:
    df = pd.read_csv(f)
    # Assume df has a column named 'val_accuracy' or similar; adjust after inspection
    if 'val_accuracy' in df.columns:
        plt.plot(df['val_accuracy'], label=f)

plt.xlabel('Epoch')
plt.ylabel('Validation accuracy')
plt.legend()
plt.show()
```

Note: column names are taken from the CSV header; inspect each CSV to find the correct metric names.

## Reproducing experiments

1. Prepare a Python environment and install dependencies as noted above.
2. Ensure the `data/` folder is reachable by `main.py` (it expects local data paths).
3. Open `main.py` and check for configurable parameters such as image size, batch size, number of epochs, and model architecture choices.
4. Run training and collect metrics. The `metrics/` folder can be used to store new CSV exports for later comparison.

## Ethical note

Research on ethnicity or demographic classification carries ethical risks (privacy, misuse, bias). Use the dataset and models responsibly, obtain consent for data usage, and consider fairness, privacy, and legal implications before deploying or publishing results.

## Future steps

- Add a `requirements.txt` or `environment.yml` to document exact dependencies.
- Add brief documentation at the top of `main.py` describing required command-line arguments and expected data layout.
- Save representative training plots to the `images/` folder and reference them here.
- Include a small script to aggregate `metrics/` CSVs into a single comparison report.

