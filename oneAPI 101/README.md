### Space Weather Severity Classification with Intel® Optimized PyTorch

This project implements a neural network to classify space weather severity using a dataset collected from the NOAA satellites.The dataset used for training the model consists of hourly observations between 2016 and 2018. Each observation includes solar wind parameters and geomagnetic field measurements.

The model is optimized using Intel® Extension for PyTorch to leverage Intel hardware for faster performance.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Training and Model Saving](#training-and-model-saving)
- [Inference](#inference)
- [License](#license)

## Overview
The goal of this project is to classify the severity of space weather conditions using a multiclass classification model. The model is trained on a dataset that includes features like `bx_gsm`, `by_gsm`, `bz_gsm`, `bt`, `intensity`, and more, and predicts a severity class from 0 to 3.

The model utilizes Intel® Extension for PyTorch to optimize the training process for Intel hardware.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8+
- PyTorch
- Intel® Extension for PyTorch (`intel_extension_for_pytorch`)
- Scikit-learn (`scikit-learn`)
- Scikit-learn-intelex (`scikit-learn-intelex`)
- pandas

You can install all the dependencies by running:

```bash
pip install pandas scikit-learn torch intel-extension-for-pytorch scikit-learn-intelex
```

## Project Structure

```bash
.
├── Dataset
│   └── spaceWeatherSeverityMulticlass.csv   # Dataset file
├── Model
│   └── spaceWeatherSeverityModel.pth        # Trained model (after running)
│   └── spaceWeatherScaler.pkl               # Saved scaler for inference
├── space_weather_severity_multiclass.py     # Main training script
├── inference.py                             # Inference script
└── README.md                                # This file
```

## How to Run

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/Events-iCGB.git
cd "oneAPI 101"
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install pandas scikit-learn torch intel-extension-for-pytorch scikit-learn-intelex
```

### 3. Run the Training Script

To train the model, simply run the `space_weather_severity_multiclass.py` script:

```bash
python space_weather_severity_multiclass.py
```

This will:
- Load the dataset (`Dataset/spaceWeatherSeverityMulticlass.csv`)
- Train the model using the Intel® Extension for PyTorch optimizations
- Save the trained model (`spaceWeatherSeverityModel.pth`) and scaler (`spaceWeatherScaler.pkl`) in the `Model/` directory

### 4. Check the Output

During training, the script will print the training loss at each epoch and the final accuracy on the test set.

Once training is complete, you will find the trained model and the scaler in the `Model/` directory.

## Training and Model Saving

- **Training Time**: The script measures the time taken for training and prints the duration at the end.
- **Saving the Model**: After training, the model is automatically saved as `spaceWeatherSeverityModel.pth` in the `Model/` folder.
- **Saving the Scaler**: The scaler used to normalize the features is saved as `spaceWeatherScaler.pkl` for consistent preprocessing during inference.

## Inference

Once the model is trained, you can use the `inference.py` script to make predictions on new data.

1. Ensure the trained model and scaler are in the `Model/` folder.
2. Run the inference script with new data:

```bash
python inference.py
```

The script will load the model and scaler, preprocess the data, and output the predictions.
