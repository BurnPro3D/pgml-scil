
# Physics-Guided ViViT for Fuel Density Prediction

This repository contains the training pipeline for a **Video Vision Transformer (ViViT)** designed to forecast spatiotemporal fuel density evolution. 

The model tokenizes input video frames into 3D spatiotemporal "tubelets" and processes them using a transformer backbone. It also integrates **Physics-Guided Machine Learning (PGML)** principles, enforcing mass conservation (fuel transport) and kinematic consistency (Rate of Spread) directly into the loss function.

## Project Structure

The training script relies on a shared `common_utils` module. Ensure your directory is structured as follows:

```text
project_root/
├── common_utils/               # Shared utilities (dataloader, loss, logging)
│   ├── config.py
│   ├── dataloader.py
│   ├── loss.py
│   └── ...
├── ViViT/                      # ViViT specific code
│   ├── configs/
│   │   ├── config_lite_data.yaml   # Config for Lite dataset
│   │   └── config_full_data.yaml   # Config for Full dataset
│   ├── src/
│   │   ├── trainer_vivit.py    # Main training entry point
│   │   └── model.py            # ViViT architecture definition
└── data/                       # Dataset directory

```

## Usage

### 1. Configuration (`config_*.yaml`)

Before running, open your desired config file (`config_lite_data.yaml` or `config_full_data.yaml`) and **update the following paths** to match your environment:

* `data_dir`: Path to your dataset (`data_lite/` or `data_lite_600_5050/`).
* `mlflow_tracking_uri`: Directory/URI for MLflow logs.
* `pretrained_path`: Set to a local path or HuggingFace ID (e.g., `"google/vivit-b-16x2-kinetics400"`) for transfer learning, or leave empty `""` to train from scratch.

### 2. Training (Single GPU)

Run the trainer script directly using Python. Point to the appropriate config file based on your dataset:

```bash
# For Lite Data Training
python ViViT/src/trainer_vivit.py --config ViViT/configs/config_lite_data.yaml

# For Full Data Training
python ViViT/src/trainer_vivit.py --config ViViT/configs/config_full_data.yaml

```

### 3. Training (Multi-GPU / DDP)

The script supports `DistributedDataParallel` (DDP). Use `torchrun` to launch it across multiple GPUs (e.g., 2 GPUs):

```bash
torchrun --nproc_per_node=2 ViViT/src/trainer_vivit.py --config ViViT/configs/config_full_data.yaml

```

---

## Configuration Guide

The YAML configuration files control all aspects of the training. Below are the key parameters and how they differ between the **Lite** and **Full** dataset training regimens.

### ViViT Architecture

These parameters define the Transformer backbone and the 3D tokenization strategy:

* `tubelet_size`: **[3, 4, 4]** *(Lite)* | **[4, 8, 8]** *(Full)*. Defines the `[time, height, width]` of each token.
* `embed_dim`: **256** *(Lite)* | **128** *(Full)*.
* `hidden_dim`: **512** *(Lite)* | **256** *(Full)*.
* `num_blocks`: **4** (Both). Number of transformer encoder layers.
* `num_attn_heads`: **4** (Both). Number of attention heads.

### Sequence & Prediction Windows

The temporal horizons differ significantly based on the dataset complexity:

* `context_len` (Input history): **6** *(Lite)* | **8** *(Full)*
* `num_pred_frames` (Prediction horizon): **3** *(Lite)* | **4** *(Full)*
* `future` (Frames to skip/predict into future): **6** *(Lite)* | **8** *(Full)*
* `temporal_stride` (Data sampling stride): **1** (Both).

### Physics-Guided Loss (`physics_loss`)

Control the physical constraints applied during training:

* `physics_loss`: **false** *(Lite)* | **true** *(Full)*. Enables/disables PGML.
* `fuel_transport_weight`: **0.01** (Penalizes unrealistic fuel regeneration).
* `burned_weight` / `unburned_weight`: **0.1** (Handles class imbalance by weighting active fire regions differently from the background).
* `fire_metrics_weight` (ROS): **0.01** (Penalizes errors in Rate of Spread).
* `mse_weight`: **1.0** (Standard MSE loss on fuel density).

### Training Hyperparameters

* `batch_size`: **4** *(Lite)* | **16** *(Full)*
* `mini_batch_size` (Gradient accumulation): **4** *(Lite)* | **2** *(Full)*
* `max_epochs`: **50** *(Lite)* | **1** *(Full)*
* `mixed_prec_dtype`: **"bfloat16"** (Highly recommended for Transformer stability).
* `max_lr`: **1.0e-3** fading to `min_lr`: **1.0e-5** (Cosine Annealing).

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
3. Calculates One-Shot MSE across the sliding windows and generates side-by-side visual animations.

