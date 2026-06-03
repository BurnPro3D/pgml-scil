import os
import sys
import math
import time
import yaml
import argparse
import numpy as np
import logging
from tqdm import tqdm
import mlflow
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile
from functools import partial
from einops import rearrange
from collections import defaultdict
from pathlib import Path

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

from common_utils import logging_utils
from FourCastNet.networks.afnonet_wf import AFNONet_Seq2Seq, AFNONet_Residual # Assumes this is in the right path
from common_utils.config import Config
from common_utils.early_stopping import EarlyStopping
from common_utils.distributed_training import setup_distributed_training
from common_utils.experiment_tracking import setup_experiment_tracking
from common_utils.dataloader import create_datasets_lite, create_dataloaders, create_datasets_full
from common_utils.dataset import DataPreprocessorLite, FullSequenceDataset, SimpleDataset
from common_utils.loss import PhysicsGuidedLoss
import cmocean.cm as cmo

RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
# Log file path will be set inside the temporary directory later

# --- Trainer Class (Handles Training and Validation) ---
class Trainer:
    def __init__(self, config, model, train_loader, val_loader, loss_fn, optimizer, scheduler, early_stop_criteria, grad_accum_steps, ddp, ddp_rank, device):
        self.config = config
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.master_process = ddp_rank == 0
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stop_criteria = early_stop_criteria
        self.max_epochs = config.max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.mixed_prec_dtype = getattr(torch, config.mixed_prec_dtype)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and config.enable_amp else None
        self.best_val_loss = float('inf')

    def train_loop(self):
        """Main training loop that handles epochs and batches"""
        step = 0  # Global step counter for the entire training
        for epoch in range(self.max_epochs):

            if epoch < 0 and self.config.physics_loss:
                self.loss_fn.use_physics_loss = False
            else:
                self.loss_fn.use_physics_loss = self.config.physics_loss
            
            logging.info(f"\n{'='*20} Starting Epoch {epoch+1}/{self.max_epochs} {'='*20}")
            self.model.train()  # Set model to training mode
            if self.ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0  # Running loss for current epoch
            epoch_loss_components = defaultdict(float)
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}", disable=not self.master_process)
            for batch_idx, (inp, tar, _) in enumerate(pbar):
                # Move input and target to device and prepare for model
                inp = inp.to(self.device, non_blocking=True)
                # tar = tar.squeeze(1).to(self.device, non_blocking=True)
                tar = tar.to(self.device, non_blocking=True)
                
                # Check if this is a gradient accumulation step
                is_accum_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == len(self.train_loader))
                eff_accum_steps = self.grad_accum_steps if not (batch_idx + 1 == len(self.train_loader)) else (batch_idx % self.grad_accum_steps + 1)
                
                # Use mixed precision training for better performance
                with torch.amp.autocast(device_type=self.device_type, dtype=self.mixed_prec_dtype):
                    pred = self.model(inp)  # Forward pass
                    # print( pred.shape, tar.shape)
                    pred = pred[:, :self.config.num_pred_frames, ...]
                    # print( pred.shape, tar.shape)
                    loss, loss_components = self.loss_fn(pred, tar)  # Calculate loss
                    loss /= eff_accum_steps  # Normalize loss for gradient accumulation
                
                self.scaler.scale(loss).backward()
                epoch_loss += loss.detach()

                # if self.config.physics_loss:
                for k, v in loss_components.items():
                    epoch_loss_components[k] += v # v is already a batch-avg float


                if is_accum_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if self.master_process:
                        # Show rolling average of components for the epoch
                        pbar_metrics = {k: v / (batch_idx + 1) for k, v in epoch_loss_components.items()}
                        pbar.set_postfix(pbar_metrics)

            if self.ddp: 
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
                for k, v in epoch_loss_components.items():
                    v_tensor = torch.tensor(v, device=self.device)
                    dist.all_reduce(v_tensor, op=dist.ReduceOp.AVG)
                    epoch_loss_components[k] = v_tensor.item()

            if self.master_process:
                avg_epoch_loss = (epoch_loss.item() / len(self.train_loader)) * self.grad_accum_steps
                mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)

                # if self.config.physics_loss:
                # Average components over all batches in the epoch
                avg_loss_components = {f'train_{k}': v / len(self.train_loader) for k, v in epoch_loss_components.items()}
                mlflow.log_metrics(avg_loss_components, step=epoch)

            val_loss = self.validate(epoch, step)
            if self.master_process:
                logging.info(f"\nEpoch {epoch+1} completed. Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            self.scheduler.step()
            if self.early_stop_criteria(val_loss):
                if self.master_process: logging.info("Early stopping triggered.")
                break

        # if self.master_process:
        #      mlflow.log_params(self.config._config)
        #      if log_filepath:
        #          mlflow.log_artifact(log_filepath, artifact_path="logs")    


    def validate(self, epoch, step):
        """Perform validation for the current epoch"""
        logging.info(f"\nRunning validation for epoch {epoch+1}...")
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        val_loss_components = defaultdict(float)
        val_steps = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for inp, tar, _ in self.val_loader:
                # Move data to device
                inp = inp.to(self.device, non_blocking=True)
                # tar = tar.squeeze(1).to(self.device, non_blocking=True)
                tar = tar.to(self.device, non_blocking=True)

                # Run model with mixed precision
                with torch.amp.autocast(device_type=self.device_type, dtype=self.mixed_prec_dtype):
                    pred = self.model(inp)  # Forward pass
                    pred = pred[:, :self.config.num_pred_frames, :, :, :]
                    loss, loss_components  = self.loss_fn(pred, tar)  # Calculate validation loss

                val_loss += loss.detach()

                # if self.config.physics_loss:
                for k, v in loss_components.items():
                    val_loss_components[k] += v  # v is already a batch-avg float
                val_steps += 1


        val_loss /= val_steps
        # if self.config.physics_loss:
        val_loss_components = {k: v / val_steps for k, v in val_loss_components.items()}

        if self.ddp: 
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            # if self.config.physics_loss:
            for k, v in val_loss_components.items():
                v_tensor = torch.tensor(v, device=self.device)
                dist.all_reduce(v_tensor, op=dist.ReduceOp.AVG)
                val_loss_components[k] = v_tensor.item()


        if self.master_process:
            val_loss_item = val_loss.item()
            logging.info(f"[VAL] epoch: {epoch+1} | val loss: {val_loss_item:.6f}")
            mlflow.log_metric("val_loss", val_loss_item, step=epoch)
            # if self.config.physics_loss:
            val_log_components = {f'val_{k}': v for k, v in val_loss_components.items()}
            mlflow.log_metrics(val_log_components, step=epoch)

            if val_loss_item < self.best_val_loss:
                self.best_val_loss = val_loss_item
                self.save_checkpoint(epoch, is_best=True)

        return val_loss.item()

    def save_checkpoint(self, epoch, is_best=False):
        raw_model = self.model.module if self.ddp else self.model
        ckpt = {'epoch': epoch, 'state_dict': raw_model.state_dict()}
        if is_best:
            ckpt_path = os.path.join(self.config.exp_dir, "best_model.pth")
            torch.save(ckpt, ckpt_path)
            logging.info(f"Saved best model checkpoint to {ckpt_path}")

# --- Testing and Visualization Functions ---

def create_comparison_animation(true_data, pred_data, save_path, title=''):
    error_data = true_data - pred_data
    vmax = 0.7 # Use a fixed vmax for consistent color scales
    error_max = np.max(np.abs(error_data))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    plt.close()
    im1 = ax1.imshow(true_data[0], cmap=cmo.thermal_r, vmin=0, vmax=vmax)
    im2 = ax2.imshow(pred_data[0], cmap=cmo.thermal_r, vmin=0, vmax=vmax)
    im3 = ax3.imshow(error_data[0], cmap=cmo.balance, vmin=-error_max, vmax=error_max)
    ax1.set_title('Ground Truth'); ax2.set_title('Prediction'); ax3.set_title('Error')
    fig.colorbar(im1, ax=ax1); fig.colorbar(im2, ax=ax2); fig.colorbar(im3, ax=ax3)
    fig.suptitle(title, fontsize=16)
    def update(frame):
        im1.set_array(true_data[frame]); im2.set_array(pred_data[frame]); im3.set_array(error_data[frame])
        fig.suptitle(f'{title}\nFrame: {frame + 1}/{len(true_data)}', fontsize=16)
        return [im1, im2, im3]
    anim = animation.FuncAnimation(fig, update, frames=len(true_data), blit=True)
    anim.save(save_path, writer='pillow', fps=10)
    logging.info(f"Saved animation to {save_path}")

# def run_autoregressive_test(model, test_loader, config, temp_run_dir):
#     """Run autoregressive testing on the model by generating multi-step forecasts"""
#     logging.info("Starting autoregressive testing...")
#     print("\n=== Beginning Autoregressive Testing ===")
#     device = next(model.parameters()).device
#     model.eval()  # Set model to evaluation mode

#     all_mses = []  # Store MSE for each test sequence
#     samples_to_visualize = config.get('test_sequences', [0, 5, 17, 22, 43, 45])  # Sample indices to create visualizations for

#     for sample_idx, (full_sequence_inputs, full_sequence_targets) in enumerate(test_loader):
#         # The loader provides a batch, so we take the first element for autoregressive testing.
#         print(full_sequence_inputs.shape, full_sequence_targets.shape)
#         full_sequence_inputs = full_sequence_inputs[0]
#         full_sequence_targets = full_sequence_targets[0]

#         logging.info(f"--- Processing Test Sample {sample_idx + 1}/{len(test_loader)} ---")
#         context_len = config.context_len
#         num_forecast_steps = len(full_sequence_inputs) - context_len
#         predictions = []
#         current_input_sequence = full_sequence_inputs[:context_len].clone().to(device)  # Initialize with context window

#         with torch.no_grad():
#             for i in tqdm(range(num_forecast_steps), desc=f"Forecasting Sample {sample_idx}"):
#                 input_for_model = current_input_sequence.unsqueeze(0)
#                 next_step_pred = model(input_for_model)
#                 predictions.append(next_step_pred.cpu().numpy())
#                 next_input_features = full_sequence_inputs[context_len + i].clone()
#                 next_input_features[-1, :, :] = next_step_pred.squeeze()
#                 new_sequence = torch.roll(current_input_sequence, shifts=-1, dims=0)
#                 new_sequence[-1, :, :, :] = next_input_features
#                 current_input_sequence = new_sequence

#         predictions = np.concatenate(predictions, axis=0).squeeze(1)
#         ground_truth = full_sequence_targets[context_len:].numpy().squeeze(1)

#         mse = mean_squared_error(ground_truth.ravel(), predictions.ravel())
#         all_mses.append(mse)
#         logging.info(f"MSE for Sample {sample_idx}: {mse:.6f}")

#         if sample_idx in samples_to_visualize:
#             anim_path = os.path.join(temp_run_dir, f"autoregressive_sample_{sample_idx}.gif")
#             create_comparison_animation(ground_truth, predictions, anim_path, title=f"Autoregressive Forecast (Sample {sample_idx})")

#     average_mse = np.mean(all_mses)
#     logging.info(f"Final Average Test MSE across all {len(test_loader)} samples: {average_mse:.6f}")
#     mlflow.log_metric("Test_Average_MSE", average_mse)

def run_seq2seq_test(model, test_loss_fn, test_loader, config, temp_run_dir):
    """
    Run one-shot sequence-to-sequence testing on the model.
    This function iterates over the test_loader, which yields
    full sequences from the SimpleDataset.
    """
    logging.info("Starting sequence-to-sequence (one-shot) testing...")
    print("\n=== Beginning Sequence-to-Sequence Test ===")
    device = next(model.parameters()).device
    model.eval()

    test_loss_fn.use_physics_loss = True

    # # if config.physics_test:
    # test_loss_fn = PhysicsGuidedLoss(
    #     use_physics_loss=config.physics_test,
    #     fuel_transport_weight=config.get("fuel_transport_weight", 0),
    #     burned_weight=config.get("burned_weight", 0),
    #     unburned_weight=config.get("unburned_weight", 0),
    #     fire_metrics_weight=config.get("fire_metrics_weight", 0),
    #     ros_method=config.get("ros_method", None),
    #     mse_weight=config.get("mse_weight", 1.0),
    # )

    # config.set_override({"fuel_transport_weight": 1.0,
    #                      "burned_weight": 1.0,
    #                      "unburned_weight": 1.0,
    #                      "fire_metrics_weight": 1.0,
    #                      "ros_method": 1.0,
    #                      "mse_weight": 1.0,
    #                     })
    test_loss_fn.physics_loss_fn.fuel_transport_weight = 1.0
    test_loss_fn.physics_loss_fn.burned_weight = 1.0
    test_loss_fn.physics_loss_fn.unburned_weight = 1.0
    test_loss_fn.physics_loss_fn.fire_metrics_weight = 1.0
    test_loss_fn.physics_loss_fn.mse_weight = 1.0

    all_mses = [] # This will store the 'base_mse'
    all_loss_components = defaultdict(list)
    samples_to_visualize = config.get('test_sequences', [0, 5, 17, 22, 43, 45])
    context_len = config.context_len
    num_pred_frames = config.num_pred_frames
    stride = 1
    future = config.get('future', context_len)

    # Get current rank for logging
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    
    with torch.no_grad():
        # Iterate over the test_loader, which yields full sequences (Batch=1)
        for sample_idx, (full_sequence_inputs, full_sequence_targets) in enumerate(test_loader):
            
            # Squeeze the batch dimension
            full_sequence_inputs = full_sequence_inputs.squeeze(0).to(device) # Shape: (T, C, H, W)
            full_sequence_targets = full_sequence_targets.squeeze(0).to(device) # Shape: (T, 1, H, W)

            logging.info(f"--- Processing Test Sample {sample_idx + 1}/{len(test_loader)} ---")

            predictions_for_viz = []
            ground_truths_for_viz = []

            # Loop through the full sequence, creating sliding windows
            # Use temporal_stride from config for the sliding window
            # stride = config.temporal_stride 
            loop_end = len(full_sequence_inputs) - future - num_pred_frames + 1
            for t in range(0, loop_end, stride):

                # 1. Get the input chunk (e.g., frames 0-4)
                input_chunk = full_sequence_inputs[t : t + context_len].unsqueeze(0) # (1, T_in, C, H, W)
                
                # 2. Get the corresponding target chunk
                # The target for this input chunk starts *after* the input
                target_start = t + future
                target_end = target_start + num_pred_frames
                target_chunk = full_sequence_targets[target_start : target_end]
                # target shape: (T_out, C_out, H, W)
                
                # 3. Predict a full sequence
                pred_seq = model(input_chunk) # Output: (1, T_in, C_out, H, W)
                
                # 4. Slice the prediction to get the frame(s) we care about
                # We slice the *start* of the prediction, matching the target
                pred_chunk = pred_seq[0, :num_pred_frames, :, :, :]
                target_chunk = target_chunk[:num_pred_frames, :, :, :]
                # pred shape: (T_out, C_out, H, W)

                # Add a batch dimension (B=1) for the loss function
                pred_tensor_b1 = pred_chunk.unsqueeze(0)
                target_tensor_b1 = target_chunk.unsqueeze(0)

                components = {}
                loss_val, components = test_loss_fn(pred_tensor_b1, target_tensor_b1, seq_idx=None)

                if 'base_mse' not in components:
                    base_mse_val = mean_squared_error(
                        target_tensor_b1.cpu().numpy().ravel(),
                        pred_tensor_b1.cpu().numpy().ravel()
                    )
                    components['base_mse'] = base_mse_val

                mask = target_tensor_b1 < 0.65
                error_sq = (pred_tensor_b1 - target_tensor_b1) ** 2
                burning_error_sq = torch.masked_select(error_sq, mask)
                
                burning_mse = 0.0
                if burning_error_sq.numel() > 0:
                    burning_mse = torch.mean(burning_error_sq).item()
                
                components['burning_mse_lt_0_65'] = burning_mse

                # Store all components
                for k, v in components.items():
                    all_loss_components[k].append(v) # v is already a float

                if sample_idx in samples_to_visualize:
                    # print(pred_chunk.shape, pred_chunk[:stride].shape)
                    predictions_for_viz.append(pred_chunk[:stride].squeeze(1).cpu())
                    ground_truths_for_viz.append(target_chunk[:stride].squeeze(1).cpu())

            
            mse = np.mean(all_loss_components['base_mse'][-(loop_end // stride):]) # Average over this sample's windows
            logging.info(f"One-shot MSE for Sample {sample_idx}: {mse:.6f}")

            if sample_idx in samples_to_visualize:
                if not predictions_for_viz:
                    logging.warning(f"No predictions generated for sample {sample_idx} for viz. Skipping.")
                    continue
                
                # Stitch chunks for *this sample's* visualization
                viz_stride = 5
                predictions_np_viz = torch.cat(predictions_for_viz, dim=0).squeeze(1).cpu().numpy()[::viz_stride]
                ground_truth_np_viz = torch.cat(ground_truths_for_viz, dim=0).squeeze(1).cpu().numpy()[::viz_stride]

                anim_path = os.path.join(temp_run_dir, f"seq2seq_test_sample_{sample_idx}.gif")
                create_comparison_animation(
                    ground_truth_np_viz, # (T, H, W)
                    predictions_np_viz, # (T, H, W)
                    anim_path, 
                    title=f"Seq-to-Seq Test (Sample {sample_idx})"
                )
            
            # if dist.is_available() and dist.is_initialized():
            #     dist.barrier()
    # average_mse = np.mean(all_mses)
    # logging.info(f"Final Average Seq-to-Seq Test MSE across all {len(test_loader)} samples: {average_mse:.6f}")
    # mlflow.log_metric("Test_Average_MSE_Seq2Seq", average_mse)


    avg_components = {}
    for k, v_list in all_loss_components.items():
        avg_val = np.mean(v_list)
        metric_name = f"Test_avg_{k}".replace('base_mse', 'MSE_Seq2Seq')
        avg_components[metric_name] = avg_val
        logging.info(f"Final Average Test {k}: {avg_val:.6f}")
        
        mlflow.log_metrics(avg_components)
# --- Main Execution Block ---

def main(args_config):
    dist_config = setup_distributed_training()
    ddp, ddp_rank, ddp_local_rank,ddp_world_size, master_process, device = dist_config.values()

    config = Config(filepath=args_config)
    config.timestamp = RUN_TIMESTAMP
    config.ddp_world_size = dist.get_world_size() if ddp else 1
    
    if master_process:
        setup_experiment_tracking(config)

    with tempfile.TemporaryDirectory() as temp_run_dir:
        if master_process:
            config.exp_dir = temp_run_dir # Use temp dir for all run outputskiuo[iko]
            logging_utils.configure_logger(RUN_TIMESTAMP, log_dir=temp_run_dir)
            logging.info(f"Run artifacts will be stored in: {temp_run_dir}")

        print(f"Data loading for DDP rank {ddp_rank}...")
        print(f"Creating datasets and dataloaders for DDP rank {ddp_rank}...")
        
        test_dataset = None
        
        if config.which_data == 'data_lite':
            train_dataset, val_dataset, test_dataset = create_datasets_lite(config, project_dir, ddp_rank, config.ddp_world_size)
            
            # Manual creation of test dataset for lite version
            # if master_process:
            #     logging.info("Creating SimpleDataset (Lite) for testing...")
            #     dataprocessor_lite = DataPreprocessorLite(project_dir, config.data_dir)
            #     _, _, test_x, test_y = dataprocessor_lite.load_data()
            #     _, _, test_x, test_y = dataprocessor_lite.preprocess_data(
            #         [], [], test_x, test_y,
            #         teacher_forcing=config.teacher_forcing,
            #         exclude_ignition=config.exclude_ignition,
            #         exclude_sourcemap=config.exclude_sourcemap,
            #         desired_height=config.image_size,
            #         desired_width=config.image_size,
            #     )
            #     test_dataset = SimpleDataset(test_x, test_y)

        elif config.which_data == 'data_full':
            train_dataset, val_dataset, _ = create_datasets_full(config, ddp_rank, config.ddp_world_size)
            
            # Manual creation of test dataset for FULL version
            if master_process:
                logging.info("Creating FullSequenceDataset (Full) for testing...")
                test_dir = os.path.join(config.data_dir, "test")
                # Limit number of files for quick testing if needed, or take all
                # test_files = test_files[:5] 
                test_dataset = FullSequenceDataset(config, test_dir)

        # Create DataLoaders
        # test_batch_size=1 ensures we get one full sequence at a time for run_seq2seq_test
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, config, test_batch_size=1
        )
        
        # Determine input channels from the first batch
        # Note: train_loader returns (x, y, idx)
        config.in_channels = next(iter(train_loader))[0].shape[2] # (B, T, C, H, W) -> C is at index 2

        # model = AFNONet3D(
        #     img_size=(config.image_size, config.image_size), patch_size=(config.patch_size, config.patch_size),
        #     in_chans=config.in_channels, out_chans=config.out_channels,
        #     context_len=config.context_len, temporal_patch_size=config.temporal_patch_size,
        #     embed_dim=config.embed_dim, depth=config.depth, num_blocks=config.num_blocks
        # ).to(device)
        model = AFNONet_Seq2Seq(
            img_size=(config.image_size, config.image_size), patch_size=(config.patch_size, config.patch_size),
            in_chans=config.in_channels, out_chans=config.out_channels,
            context_len=config.context_len, temporal_patch_size=config.temporal_patch_size,
            embed_dim=config.embed_dim, depth=config.depth, num_blocks=config.num_blocks
        ).to(device)
        
        # model = AFNONet_Residual(backbone).to(device)
        if config.torch_compile: model = torch.compile(model)
        if ddp: model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
        
        loss_fn = PhysicsGuidedLoss(
            use_physics_loss=config.physics_loss,
            fuel_transport_weight=config.get("fuel_transport_weight", 0),
            burned_weight=config.get("burned_weight", 0),
            unburned_weight=config.get("unburned_weight", 0),
            fire_metrics_weight=config.get("fire_metrics_weight", 0),
            ros_method=config.get("ros_method", None),
            mse_weight=config.get("mse_weight", 1.0),
        )        
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.max_lr, momentum=0.9, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=config.min_lr)
        early_stop_criteria = EarlyStopping(patience=config.early_stopping_patience)
        grad_accum_steps = config.batch_size // (config.mini_batch_size * config.ddp_world_size)

        trainer = Trainer(
            config, model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            early_stop_criteria, grad_accum_steps, ddp, ddp_rank, device
        )
        trainer.train_loop()

        # 1. Sync before testing starts (ensures all ranks finished training)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            print(f"Rank {ddp_rank}: Training finished. Sync complete.")

        if ddp_rank != 0:
            print(f"Rank {ddp_rank}: Exiting process. Leaving testing to Rank 0.")
            # Optional: Destroy process group to be clean, though usually script exit handles it
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        # We want Rank 0 to see ALL 35 files, not just 1/4th of them.
        if config.which_data == "data_full":
            # We don't need the train/val datasets anymore, just Test
            logging.info("Creating FullSequenceDataset for centralized testing...")
            test_dir = os.path.join(config.data_dir, "test")
            test_dataset = FullSequenceDataset(config, test_dir)
            
            # Create standard DataLoader (No Sampler = Sequential full access)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=1, # One sequence at a time
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        
        # # If model is DDP wrapped, access the underlying module to avoid DDP overhead/errors
        # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #     inference_model = model.module
        # else:
        #     inference_model = model

        # --- TESTING STAGE ---
        if master_process:
            logging.info("\n--- TRAINING COMPLETE. STARTING FINAL EVALUATION. ---")
            # Load the best model for testing
            best_model_path = os.path.join(temp_run_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                raw_model = model.module if ddp else model
                checkpoint = torch.load(best_model_path, map_location=device)
                raw_model.load_state_dict(checkpoint['state_dict'])
                logging.info(f"Successfully loaded best model from epoch {checkpoint['epoch']} for testing.")

                # Run autoregressive test
                # run_autoregressive_test(raw_model, test_loader, config, temp_run_dir)
                run_seq2seq_test(raw_model, loss_fn, test_loader, config, temp_run_dir)
            else:
                logging.warning("Could not find best_model.pth to run final evaluation.")

            # Log all artifacts from the temporary directory to MLflow
            logging.info("Logging all run artifacts to MLflow...")
            mlflow.log_artifacts(temp_run_dir, artifact_path="run_outputs")

        # 2. Sync after testing (Non-master ranks wait here while Master tests)
        # This prevents the job from terminating early.
        # Cleanup Rank 0
        if dist.is_initialized():
            dist.destroy_process_group()
        # if ddp: dist.barrier()

        # if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="/home/pgmlvol/tcaglar/pgml/FourCastNet/configs/config_lite_data.yaml", help="Path to the config file")
    args, _ = parser.parse_known_args()

    main(args.config)