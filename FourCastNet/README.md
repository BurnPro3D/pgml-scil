# Physics-Guided AFNONet for Fuel Density Prediction

This repository contains the training pipeline for **AFNONet (Adaptive Fourier Neural Operator Network)**, adapted from the FourCastNet architecture, to forecast spatiotemporal fuel density evolution in prescribed fires.

The model leverages **Physics-Guided Machine Learning (PGML)**, integrating a `PhysicsGuidedLoss` that enforces mass conservation (fuel transport) and kinematic consistency (Rate of Spread) alongside standard reconstruction loss.

## Project Structure

The training script relies on a shared `common_utils` module and the `FourCastNet` package structure. Ensure your directory is organized as follows:

```text
project_root/
├── common_utils/               # Shared utilities (dataloader, loss, logging)
│   ├── config.py
│   ├── dataloader.py
│   └── ...
├── FourCastNet/                # AFNO specific code
│   ├── configs/
│   │   ├── config_lite_data.yaml   # Config for Lite dataset
│   │   └── config_full_data.yaml   # Config for Full dataset
│   ├── networks/
│   │   └── afnonet_wf.py       # AFNONet_Seq2Seq model definition
│   └── src/
│       └── trainer_new.py      # Main training entry point
└── data/                       # Dataset directory

```

## Usage

### 1. Configuration (`config_*.yaml`)

Before running, open your desired config file (`config_lite_data.yaml` or `config_full_data.yaml`) and **update the following paths** to match your environment:

* `data_dir`: Path to your dataset (`data_lite/` or `data_lite_600_5050/`).
* `exp_dir`: Where results and checkpoints will be saved.
* `mlflow_tracking_uri`: Directory/URI for MLflow logs.

### 2. Training (Single GPU)

Run the trainer script directly using Python. Point to the appropriate config file based on your dataset:

```bash
# For Lite Data Training
python FourCastNet/src/trainer_new.py --config FourCastNet/configs/config_lite_data.yaml

# For Full Data Training
python FourCastNet/src/trainer_new.py --config FourCastNet/configs/config_full_data.yaml

```

### 3. Training (Multi-GPU / DDP)

The script supports `DistributedDataParallel` (DDP). Use `torchrun` to launch it across multiple GPUs (e.g., 2 GPUs):

```bash
torchrun --nproc_per_node=2 FourCastNet/src/trainer_new.py --config FourCastNet/configs/config_full_data.yaml

```

---

## Configuration Guide

The YAML configuration files control the AFNO architecture and training dynamics. Below are the key parameters and how they differ between the **Lite** and **Full** dataset training regimens.

### AFNO Architecture (Backbone)

These parameters define the Fourier Mixing Transformer backbone. They are largely consistent across both datasets:

* `image_size`: **296** (Fixed resolution).
* `patch_size`: **8** (Tokenization grid size).
* `embed_dim`: **256** (Dimensionality of token embeddings).
* `depth`: **4** (Number of Transformer layers).
* `num_blocks`: **4** (Number of spatial frequency blocks for mixing).
* `temporal_patch_size`: **3** *(Lite)* | **4** *(Full)* (Tubelet time dimension).

### Sequence & Prediction Windows

The temporal horizons differ significantly based on the dataset complexity:

* `context_len` (Input history): **6** *(Lite)* | **8** *(Full)*
* `num_pred_frames` (Prediction horizon): **3** *(Lite)* | **4** *(Full)*
* `future` (Frames to skip/predict into future): **6** *(Lite)* | **8** *(Full)*
* `temporal_stride` (Data sampling stride): **1** *(Lite)* | **2** *(Full)*

### Physics-Guided Loss (`physics_loss: true`)

The physics constraints are weighted differently to balance gradients in the larger dataset:

* `fuel_transport_weight`: **1.0** *(Lite)* | **1.0** *(Full)*
* `burned_weight`: **1.0** *(Lite)* | **1.0** *(Full)*
* `unburned_weight`: **1.0** *(Lite)* | **0.1** *(Full)*
* `fire_metrics_weight` (ROS): **1.0** *(Lite)* | **0.1** *(Full)*
* `ros_method`: **"horizontal_average"** *(Lite)* | **"horizontal_instant"** *(Full)*

### Training Hyperparameters

* `batch_size`: **8** *(Lite)* | **16** *(Full)*
* `mini_batch_size` (Gradient accumulation): **8** *(Lite)* | **4** *(Full)*
* `max_epochs`: **50** (Both).
* `mixed_prec_dtype`: **"bfloat16"** (Crucial for Transformer stability).
* `max_lr`: **1.0e-3** fading to `min_lr`: **1.0e-5** (Cosine Annealing).

---

## Outputs & Logging

The trainer automatically logs metrics and artifacts to **MLflow**.

### 1. Metrics

* `train_loss`, `val_loss`: Convergence curves.
* `val_base_mse`: Standard reconstruction error.
* `val_burning_mse`: Error specifically in active fire regions (<0.65 density).
* Physics Components: `fuel_loss`, `ros_loss`, `burned_loss`, `unburned_loss`.

### 2. Artifacts

Upon completion, the following are saved in the `run_outputs` artifact directory in MLflow:

* `best_model.pth`: Checkpoint with the lowest validation loss.
* **Test Animations (`.gif`)**: Visual comparisons of Ground Truth vs. Prediction vs. Error for specific test sequences.

## Testing Phase

After training, the script triggers a **Sequence-to-Sequence Test**:

1. Loads `best_model.pth`.
2. Runs inference on defined `test_sequences` (e.g., sequences `[5, 17, 22, 43, 45]`).
3. Calculates **One-Shot MSE** for sequence predictions and generates side-by-side visual animations.


