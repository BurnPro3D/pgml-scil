
# Physics-Guided ConvLSTM for Fuel Density Prediction

This repository contains the training pipeline for a **Physics-Enhanced Convolutional LSTM (ConvLSTM)** designed to forecast spatiotemporal fuel density evolution. 

The model utilizes **Physics-Guided Machine Learning (PGML)** principles, enforcing physical constraints such as mass conservation (fuel transport) and kinematic consistency (Rate of Spread) directly into the loss function. It also employs **Deep Supervision** (computing loss on intermediate layers) to stabilize training.

## Project Structure

The training script relies on a shared `common_utils` module. Ensure your directory is structured as follows:

```text
project_root/
├── common_utils/               # Shared utilities (dataloader, loss, logging)
│   ├── config.py
│   ├── dataloader.py
│   ├── loss.py
│   └── ...
├── convlstm_new/               # ConvLSTM specific code
│   ├── config_lite_data.yaml   # Config for Lite dataset
│   ├── config_full_data.yaml   # Config for Full dataset
│   ├── trainer_new.py          # Main training entry point
│   └── model.py                # Model architecture definition
└── data/                       # Dataset directory

```

## Usage

### 1. Configuration (`config_*.yaml`)

Before running, open your desired config file (`config_data_lite.yaml` or `config_full_data.yaml`) and **update the following paths** to match your environment:

* `data_dir`: Path to your dataset (`data_lite/` or `data_full/`).
* `exp_dir`: Where results and checkpoints will be saved.
* `mlflow_tracking_uri`: Directory/URI for MLflow logs.

### 2. Training (Single GPU)

Run the trainer script directly using Python. Point to the appropriate config file based on your dataset:

```bash
# For Lite Data Training
python convlstm_new/trainer.py --config convlstm_new/config_data_lite.yaml

# For Full Data Training
python convlstm_new/trainer.py --config convlstm_new/config_full_data.yaml

```

### 3. Training (Multi-GPU / DDP)

The script supports `DistributedDataParallel` (DDP). Use `torchrun` to launch it across multiple GPUs (e.g., 2 GPUs):

```bash
torchrun --nproc_per_node=2 convlstm_new/trainer.py --config convlstm_new/config_full_data.yaml

```

---

## Configuration Guide

The YAML configuration files control all aspects of the training. Below are the key parameters and how they differ between the **Lite** and **Full** dataset training regimens.

### Model Architecture & Sequence Windows

These parameters define the temporal forecasting horizons and spatial resolutions. They remain consistent across both datasets:

* `image_size`: **296** (Fixed resolution).
* `context_len`: **6** (Number of past frames input to the model).
* `num_pred_frames`: **3** (Number of future frames to predict).
* `future`: **6** (Number of frames into the future to predict).
* `temporal_stride`: **1** (Temporal sampling interval).

### Physics-Guided Loss (`physics_loss: true`)

The physical constraints applied during training. For the ConvLSTM, these are kept identical across both datasets:

* `fuel_transport_weight`: **0.01** (Penalizes unrealistic fuel regeneration).
* `burned_weight`: **0.1** (Penalizes error in burned cell regions).
* `unburned_weight`: **0.1** (Penalizes error in unburned cell regions).
* `fire_metrics_weight` (ROS): **0.01** (Penalizes errors in Rate of Spread).
* `mse_weight`: **1.0** (Standard MSE loss on fuel density).
* `ros_method`: **"horizontal_average"** (Method for calculating Rate of Spread).

### Training Hyperparameters

Due to the different sizes of the datasets, the training dynamics differ:

* `batch_size`: **2** *(Lite)* | **8** *(Full)*
* `mini_batch_size` (Gradient accumulation): **2** *(Lite)* | **2** *(Full)*
* `max_epochs`: **50** *(Lite)* | **15** *(Full)*
* `early_stopping_patience`: **10** (Both).
* `max_lr`: **1.0e-3** fading to `min_lr`: **1.0e-5** (Cosine Annealing).
* `mixed_prec_dtype`: **"bfloat16"** (Improves speed and reduces memory usage).

---

## Outputs & Logging

The trainer automatically logs metrics and artifacts to **MLflow**.

### 1. Metrics

* `train_loss`, `val_loss`: Overall loss curves.
* `val_base_mse`: Standard reconstruction error.
* `val_burning_mse`: Error specifically in active fire regions (<0.65 density).
* Physics components: `fuel_loss`, `ros_loss`, `burned_loss`, `unburned_loss`.

### 2. Artifacts

Upon completion, the following are saved in the `run_outputs` artifact directory in MLflow:

* `best_model.pth`: Checkpoint with the lowest validation loss.
* **Test Animations (`.gif`)**: Visual comparisons of Ground Truth vs. Prediction vs. Error for specific test sequences.

## Testing Phase

After the training loop finishes, the script automatically triggers a **Sequence-to-Sequence Test**:

1. Loads the `best_model.pth`.
2. Runs inference on defined `test_sequences` (e.g., `[5, 17, 22, 43, 45]`).
3. Calculates One-Shot MSE and generates side-by-side visual animations.

