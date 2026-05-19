# Introduction

This repository contains the code used for the submission "Physics-Guided Emulation for Simulating Prescribed Fires". This project relates to the problem of fuel density prediction for prescribed fires.

# Getting Started

We follow the guidelines below to train and evaluate the model.

### Dependencies

The code is implemented using Tensorflow 2.0 and NVIDIA A40 GPU. The training time is 5 hours. Training the model without the GPU requires ~8 hours.

### Installing

In order to run the code, make sure all the libraries from the requirements.txt file are installed in your conda environment.


# Setup

### Modeling Setup

The dataset consists of 50 training examples and 50 test samples. The initian wind speed and wind direction values are scalars that are repeated throughout the grid and sequence to match the other input tensors. Wind speed can vary from 1 m/s to 15 m/s. Wind direction varies from 230 $^\circ$ to 330 $^\circ$. Source and target fuel densities vary between 0 and 0.7 kg / m^2. Each time step consists of a spatial grid with 300 x 300 cells where each cell has spatial resolution of 2 m x 2 m. We use min max scaling to pre-process the wind data.


# Experimental Setup


We use simulation runs from the QUIC-Fire model \footnote[1]{Simulation runs can be generated using the code provided in \href{https://github.com/QUIC-Fire-TT/ttrs_quicfire/tree/main}{this GitHub repository}} to learn the prescribed fire emulator. To test generalization under different environmental factors, the simulation runs include 5 different ignition patterns, 7 wind speeds, and 11 wind directions. We simulate and use 100 runs as training examples. The simulation runs are for a grassland setting with two-dimensional evolution of fires captured in 300 x 300 cells grid over $n$ time steps at 1 second time intervals. Each cell is at a 2m x 2m resolution. In the experimental setup, we randomly split the 100 runs and put 50\% of the data into training and the rest into test dataset, with each comprising 50 samples. Therefore, each of the datasets has input data with dimensionality $50 \text{ simulation runs } \times 50 \text{ time steps }\times 300 \text{ rows } \times 300 \text{ columns } \times 4 \text{ features }$. Wind data is standardized using min-max scaling, whereas fuel density data is not scaled since it varies between 0 and 0.7. With batch size 1 and using the Adam optimization method for gradient estimation, we train each model for 250 epochs \footnote[1]{Code link provided \href{https://drive.google.com/drive/folders/1kRRH_7an68mQGHUOcbilwupTdn5wyjhX?usp=sharing}{here}}. The code is implemented using Tensorflow 2.0 and NVIDIA A40 GPU. Hyperparameters, including penalty coefficients in the loss terms, are fine-tuned using random grid search in the models. Learning rate is 0.001, $\lambda_{FT}, \lambda_{Burned} = 0.001$, $\lambda_{Unburned}, \lambda_{FM} = 0.0001$. In the experiments, we use $\epsilon=$ 0.001 for the physical constraint loss masking. We also use $\epsilon_b=$0.1 and $\epsilon_u=$0.65 for the burned and unburned loss masking, respectively. In the generalization results, we sample test runs into different datasets with different physical properties. This includes sampling based on wind speed, wind direction, and ignition patterns. $\mathcal{D}_{\text{Low Wind}}$ dataset has samples with initial wind speed less than 10 m/s. $\mathcal{D}_{\text{High Wind}}$ dataset has samples with initial wind speed greater than 10 m/s. $\mathcal{D}_{\text{NW Wind}}$ dataset has samples with initial wind direction blowing from the northwest, and $\mathcal{D}_{\text{SW Wind}}$ dataset has samples with initial wind direction originating from the southwest direction. $\mathcal{D}_{\text{Aerial}}$, $\mathcal{D}_{\text{Outward}}$, $\mathcal{D}_{\text{Strip South}}$, $\mathcal{D}_{\text{Inward}}$ and $\mathcal{D}_{\text{Strip North}}$ datasets include samples with different ignition patterns for igniting the fire. 

- Hyper-parameters were fine-tuned using random grid search. Physics guided loss term penalty coefficients were selected from the grid {10.0, 2.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001}. Similarly, learning rate was fine tuned from the grid {0.03, 0.01, 0.001, 0.0001, 0.0003, 0.0004, 0.0005, 0.0008, 0.00005, 0.00001 }.

- The average MSEs that are reported are from models that are trained and tested with 10 repetitions.

- As mentioned in the main text, we evaluate several metrics that compute how well the model predicts the evolution of fire over time. In the tables, the downward arrow indicated that lower values of evaluation metrics indicate better model performance. We evaluate MSE, burned area MSE, unburned area MSE and fire metric MSE (ROS MSE + BA MSE) on test set. We formulate a metric, \textbf{Dynamic MSE (DMSE)}, that evaluates model performance based on change in fuel density over time. For time steps with bigger change in the observed fuel density, we ensure that the error in predictions are penalized more, $ DMSE = {(\sum_N (Y_t - Y_{t-1}) \cdot (||\hat{Y}_t - Y_t||) )}/{( \sum_N (Y_t - Y_{t-1}))}$. We can further validate the physical consistency of the predictions by evaluating how often the physical constraints are met in the predicted values using the following metrics. \textbf{Metric}$_{\text{FT}}$ is the percentage of cells that do not follow the fuel transport constraint in the predicted values. \textbf{Metric}$_{\text{Burned}}$ is the percentage of unburned cells that are predicted to be burned. \textbf{Metric}$_{\text{Unburned}}$ is the percentage of unburned cells where fuel densities are underestimated. \textbf{Metric}$_{\text{False Positive}}$ is the percentage of burning cells that are predicted to be burned. \textbf{Metric}$_{\text{False Negative}}$ is the percentage of burning cells that are predicted to be unburned.

- For the data sparcity experiments, we report the mean and standard error in MSE values over all test samples.

- Average training time for the physics-guided model is 5 hours. Average model inference time is 24 seconds.

- We use NVIDIA A40 GPU for training and Tensorflow for building the model.

# Build and Test

Model can be build using the SOURCE/TRAIN/train.py file and evaluated using the SOURCE/TEST/test.py file. Files SOURCE/TRAIN/models.py and SOURCE/TRAIN/losses.py outline the alternate models that can also be used. In the current format, train.py implements the PGCL model.

# Outline

ICDM_PrescribedFires

│   README.txt

└───SOURCE

│   │   TEST

│   │      test.py

│   │      plotting.ipynb

│   │   TRAIN

│   │      train.py

│   │   FIGURES

│   │      PGCLplus.png

│   │      ...

│   

└───DATA

│   │   RESULTS

│   │   ...




# References

QUIC-Fire model for data generation:

Linn, R. R., Goodrick, S. L., Brambilla, S., Brown, M. J., Middleton, R. S., O'Brien, J. J., & Hiers, J. K. (2020). QUIC-fire: A fast-running simulation tool for prescribed fire planning. Environmental Modelling & Software, 125, 104616.



