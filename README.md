# Basket Ball Shot Classification 
### Shoot / No Shoot
Action Recognition Using Sensor Data with LSTM and CNN
More Actions can be added by making some changes as per requirement only shoot or no shoot was required 
---

# Action Recognition Using Sensor Data with LSTM and CNN

This repository contains the research and implementation for detecting shooting actions from sensor data. The project involves data fetching, preprocessing, deep learning model training, evaluation, and inference. The goal is to distinguish between "Shoot" and "No Shoot" actions using data collected from multiple text files.

## Project Overview

The code in this project performs the following tasks:

1. **Data Fetching & Preprocessing**
   - Dataset (https://archive.ics.uci.edu/dataset/587/basketball+dataset)
   - Upload your dataset and mount the drive with this data. 

2. **Model Preprocessing & Training**
   - Encodes the target variable and splits the data into training and testing sets.
   - Constructs a deep learning model that starts with a 1D convolutional layer for local feature extraction followed by a Masking layer, an LSTM layer to capture temporal dependencies, a Dropout layer for regularization, and finally Dense layers for classification.
   - Compiles the model using the Adam optimizer and sparse categorical crossentropy as the loss function.
   - Trains the model and records the training history (loss and accuracy).

3. **Evaluation and Inference**
   - Evaluates the trained model on a test set and prints the accuracy.
   - Generates plots for loss and accuracy over epochs.
   - Displays a confusion matrix to visualize the performance.
   - Uses the trained model to predict on new sample data for both "Shoot" and "No Shoot" scenarios.

## Data Preprocessing

- **Data Sources:** Multiple text files stored in Google Drive, containing sensor readings such as acceleration (X, Y, Z), resultant force (R), and angles (Theta, Phi).
- **Labeling:** Two groups of files are used:
  - **Shoot Data:** Labeled as "Yes".
  - **No Shoot Data:** Labeled as "No".
- **Sequence Generation:** A sliding window (with a sequence length of 3 and a step of 1) is applied to the sensor data to create sequences suitable for time series input into the model.

## Model Architecture

The deep learning model is implemented using TensorFlow Keras and includes:

- **Conv1D Layer:** Extracts local features from the time-series data.
- **MaxPooling1D (Optional):** Reduces dimensionality and emphasizes the most prominent features.
- **Masking Layer:** Handles padded sequences (masking zero values).
- **LSTM Layer:** Captures temporal dependencies across the sequence.
- **Dropout Layer:** Prevents overfitting by randomly setting a fraction of input units to zero during training.
- **Dense Layers:** The final layers perform non-linear transformations, with the last Dense layer using a softmax activation to output class probabilities.

The model is compiled with the Adam optimizer and uses sparse categorical crossentropy as the loss function.

## Training

- **Data Splitting:** The data is split into training (70%) and testing (30%) sets.
- **Training Parameters:** The model is trained for 20 epochs with a batch size of 32.
- **Monitoring:** Training history (loss and accuracy) is recorded and plotted to monitor the model’s performance over epochs.

## Evaluation

- **Test Evaluation:** The model’s accuracy is evaluated on the test set.
- **Visualizations:** Loss and accuracy plots are generated to illustrate training progress.
- **Confusion Matrix:** A confusion matrix is plotted to display the classification performance for each class.

## Inference

After training, the model is used for inference on new data:

- **No Shoot Example:** New sensor data is processed into sequences and the model predicts whether each sequence corresponds to a "No Shoot" action.
- **Shoot Example:** Similarly, new data for a shooting scenario is fed into the model and predictions are made.

- Results may look like this
- ![image](https://github.com/user-attachments/assets/a9805889-c486-4c21-82ef-22b07a945405)
- Improving the technique

- Loss over epochs
- ![image](https://github.com/user-attachments/assets/02483922-e216-4adb-a674-591eaff561e6)
- Accuracy over epochs
- ![image](https://github.com/user-attachments/assets/12d440f7-554d-4d96-8b34-b8f9f686d54e)





## Dependencies

- **Python 3.x**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **TensorFlow (Keras)**
- **Matplotlib**
