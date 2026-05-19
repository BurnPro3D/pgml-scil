import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

# Add project directory to path
project_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(project_dir)

import common_utils.physics_loss_helpers as plh


# class WildFirePhysicsLoss:
#     def __init__(
#         self, 
#         reduction='none', 
#         fuel_transport_weight=0, 
#         burned_weight=0, 
#         unburned_weight=0, 
#         fire_metrics_weight=0, 
#         ros_method=None,
#         mse_weight=1.0, 
#     ):
#         self.cell_size = (2.0, 2.0)
#         self.temporal_resolution = 1.0
#         self.ros_interval = 1
#         self.ros_sliding_window = True
#         self.ros_method = ros_method
        
#         self.fuel_transport_weight = fuel_transport_weight
#         self.burned_weight = burned_weight
#         self.unburned_weight = unburned_weight
#         self.fire_metrics_weight = fire_metrics_weight
#         self.mse_weight = mse_weight
#         self.reduction = reduction

#     def __call__(self, y_true, y_pred, sample_weight=None):
#         """
#         Calculate loss with additional Hard-Threshold Weighted MSE components.
#         """
#         # Base MSE loss map (Batch, Time, 1, H, W)
#         loss_map = (y_true - y_pred)**2 
        
#         # 1. Base Global MSE
#         base_mse_loss = self.mse_weight * torch.mean(loss_map)
#         loss_final = base_mse_loss

#         loss_components = {
#             'base_mse': base_mse_loss.detach().cpu().item(),
#             'fuel_transport': 0.0,
#             'burned': 0.0,
#             'unburned': 0.0,
#             'fire_metrics': 0.0
#         }
        
#         # 2. Fuel Transport Loss (Conservation of Mass)
#         if y_pred.size(1) > 1 and self.fuel_transport_weight > 0:
#             # Enforce that fuel cannot increase over time (monotonic decrease)
#             # (t+1) - (t) should be <= 0. If > 0, it's a violation.
#             fuel_increase = y_pred[:, 1:, ...] - y_pred[:, :-1, ...]
#             mask_violation = fuel_increase > 0
            
#             # Select only the pixels that violated physics
#             fuel_transport_diff = torch.masked_select(fuel_increase**2, mask_violation)
            
#             if fuel_transport_diff.numel() > 0:
#                 fuel_transport_loss = self.fuel_transport_weight * torch.mean(fuel_transport_diff)
#                 loss_final += fuel_transport_loss
#                 loss_components['fuel_transport'] = fuel_transport_loss.detach().cpu().item()
        
#         # --- NEW: Hard Threshold Weighted MSE ---
        
#         # 3. Burned Region Loss (Hard Threshold <= 0.1)
#         # We penalize errors strictly in regions where fuel has been consumed.
#         if self.burned_weight > 0:
#             # Create Boolean Mask: True where fuel density is <= 0.1 (Burned/Ash)
#             mask_burned = y_true <= 0.65
            
#             # Extract MSE errors only for these pixels
#             burned_errors = torch.masked_select(loss_map, mask_burned)
            
#             # Avoid NaN if no pixels are burned in this batch
#             if burned_errors.numel() > 0:
#                 burned_loss = self.burned_weight * torch.mean(burned_errors)
#                 loss_final += burned_loss
#                 loss_components['burned'] = burned_loss.detach().cpu().item()

#         # 4. Unburned Region Loss (Hard Threshold >= 0.65)
#         # We penalize errors strictly in regions where fuel is intact.
#         if self.unburned_weight > 0:
#             # Create Boolean Mask: True where fuel density is >= 0.65 (Unburned/Green)
#             mask_unburned = y_true >= 0.65
            
#             # Extract MSE errors only for these pixels
#             unburned_errors = torch.masked_select(loss_map, mask_unburned)
            
#             if unburned_errors.numel() > 0:
#                 unburned_loss = self.unburned_weight * torch.mean(unburned_errors)
#                 loss_final += unburned_loss
#                 loss_components['unburned'] = unburned_loss.detach().cpu().item()

#         # ----------------------------------------

#         # 5. Fire metrics loss (ROS/Perimeter)
#         if self.fire_metrics_weight > 0 and y_pred.size(1) > 1:
#             # Note: This requires your helper library 'plh' to be imported in the file
#             # or passed to the class. Ensure plh.calculate_fire_ros is available.
            
#             # Calculate Rate of Spread (ROS) metrics
#             try:
#                 true_timesteps, true_ros = plh.calculate_fire_ros(
#                     y_true, 
#                     cell_size=self.cell_size, 
#                     temporal_resolution=self.temporal_resolution,
#                     ros_method=self.ros_method,
#                     interval=self.ros_interval,
#                     sliding_window=self.ros_sliding_window,
#                 )
#                 pred_timesteps, pred_ros = plh.calculate_fire_ros(
#                     y_pred, 
#                     cell_size=self.cell_size, 
#                     temporal_resolution=self.temporal_resolution,
#                     ros_method=self.ros_method,
#                     interval=self.ros_interval,
#                     sliding_window=self.ros_sliding_window,
#                 )
                
#                 ros_loss_arr = (true_ros - pred_ros)**2
#                 if torch.numel(ros_loss_arr) > 0:
#                     ros_loss = self.fire_metrics_weight * torch.mean(ros_loss_arr)
#                     loss_final += ros_loss
#                     loss_components['fire_metrics'] = ros_loss.detach().cpu().item()
#             except NameError:
#                 # Handle case where plh is not defined in this scope
#                 pass
        
#         return loss_final, loss_components

class WildFirePhysicsLoss:
    def __init__(
        self, 
        reduction='none', 
        fuel_transport_weight=0, 
        burned_weight=0, 
        unburned_weight=0, 
        fire_metrics_weight=0, 
        ros_method=None,
        mse_weight=1.0, 
    ):
        self.cell_size = (2.0, 2.0)
        self.temporal_resolution = 1.0
        self.ros_interval = 1
        self.ros_sliding_window = True
        self.ros_method = ros_method
        
        self.fuel_transport_weight = fuel_transport_weight
        self.burned_weight = burned_weight
        self.unburned_weight = unburned_weight
        self.fire_metrics_weight = fire_metrics_weight
        self.mse_weight = mse_weight
        self.reduction = reduction

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Calculate the combined physics-guided loss using differentiable masks.
        """
        # Base MSE loss
        # Element-wise squared differences (Batch, Time, 1, H, W)
        loss_map = (y_true - y_pred)**2 
        
        base_mse_loss = self.mse_weight * torch.mean(loss_map)
        loss_final = base_mse_loss

        loss_components = {
            'base_mse': base_mse_loss.detach().cpu().item(),
            'fuel_transport': 0.0,
            'burned': 0.0,
            'unburned': 0.0,
            'fire_metrics': 0.0
        }
        
        # 1. Fuel Transport Loss
        # (Kept as is, using the hard mask for violations, which is differentiable enough)
        if y_pred.size(1) > 1 and self.fuel_transport_weight > 0:
            mask = (y_pred[:, 1:, ...] - y_pred[:, :-1, ...]) > 0
            fuel_transport_diff = torch.masked_select(loss_map[:, 1:, ...], mask)
            if torch.numel(fuel_transport_diff) > 0:
                fuel_transport_loss = self.fuel_transport_weight * torch.mean(fuel_transport_diff)
                loss_final += fuel_transport_loss
                loss_components['fuel_transport'] = fuel_transport_loss.detach().cpu().item()
        
        burn_prob_true = plh.fueldens_to_burnindex_differentiable(y_true)  # Differentiable burned probability map
        
       # 2. Burned Area Loss (Per-Sample Averaging)
        if self.burned_weight > 0:
            # weighted_loss shape: (B, T, 1, H, W)
            weighted_loss = loss_map * burn_prob_true
            
            # Sum over all dimensions EXCEPT Batch (dim 0)
            # Result shape: (B, T,)
            sample_loss_sum = torch.sum(weighted_loss, dim=[2, 3, 4])
            sample_weight_sum = torch.sum(burn_prob_true, dim=[2, 3, 4])
            
            # Safety: Avoid division by zero for samples with NO fire
            # We create a mask for valid samples (weight > epsilon)
            valid_sample_mask = sample_weight_sum > 1e-6
            
            # Calculate mean only where weight > 0
            # If weight is 0, the result is 0 (handled by the multiplication below)
            safe_denominator = sample_weight_sum + 1e-6
            sample_means = sample_loss_sum / safe_denominator
            
            # Only count valid samples in the final batch average
            # This ensures we don't dilute the loss with zeros from empty samples
            if torch.any(valid_sample_mask):
                # Select only the valid means
                valid_means = torch.masked_select(sample_means, valid_sample_mask)
                burned_loss = self.burned_weight * torch.mean(valid_means)
            else:
                burned_loss = torch.tensor(0.0, device=y_pred.device)
            
            loss_final += burned_loss
            loss_components['burned'] = burned_loss.detach().cpu().item()

        # 3. Unburned Area Loss (Per-Sample Averaging)
        if self.unburned_weight > 0:
            # The "unburned probability" is simply (1.0 - burn_prob_true)
            unburn_prob_true = 1.0 - burn_prob_true
            
            # weighted_loss shape: (B, T, 1, H, W)
            weighted_loss_unburned = loss_map * unburn_prob_true
            
            # Sum over all dimensions EXCEPT Batch (dim 0)
            # Result shape: (B, T,)
            sample_loss_sum_unburned = torch.sum(weighted_loss_unburned, dim=[2, 3, 4])
            sample_weight_sum_unburned = torch.sum(unburn_prob_true, dim=[2, 3, 4])
            
            # Safety: Avoid division by zero (e.g., if a sample is 100% burned)
            valid_sample_mask_unburned = sample_weight_sum_unburned > 1e-6
            
            # Calculate mean only where weight > 0
            safe_denominator_unburned = sample_weight_sum_unburned + 1e-6
            sample_means_unburned = sample_loss_sum_unburned / safe_denominator_unburned
            
            # Only count valid samples in the final batch average
            if torch.any(valid_sample_mask_unburned):
                valid_means_unburned = torch.masked_select(sample_means_unburned, valid_sample_mask_unburned)
                unburned_loss = self.unburned_weight * torch.mean(valid_means_unburned)
            else:
                unburned_loss = torch.tensor(0.0, device=y_pred.device)
            
            loss_final += unburned_loss
            loss_components['unburned'] = unburned_loss.detach().cpu().item()
        
        # 4. Fire metrics loss (ROS)
        # Note: This part still relies on 'plh' which needs to be fully differentiable 
        # as discussed previously (using soft counting/centroids).
        if self.fire_metrics_weight > 0 and y_pred.size(1) > 1:
            # Calculate Rate of Spread (ROS) and Burned Area (BA) metrics
            true_burned_frac = plh.calculate_burned_area(y_true, cell_size=self.cell_size)  # Metrics from ground truth
            pred_burned_frac = plh.calculate_burned_area(y_pred, cell_size=self.cell_size)  # Metrics from predictions
            ba_loss_arr = (true_burned_frac - pred_burned_frac)**2
            if torch.numel(ba_loss_arr) > 0:
                ba_loss = self.fire_metrics_weight * torch.mean(ba_loss_arr)
                loss_final += ba_loss
                loss_components['fire_metrics'] += ba_loss.detach().cpu().item()

            true_timesteps, true_ros = plh.calculate_fire_ros(
                y_true, 
                cell_size=self.cell_size, 
                temporal_resolution=self.temporal_resolution,
                ros_method=self.ros_method,
                interval=self.ros_interval,
                sliding_window=self.ros_sliding_window,
            )
            pred_timesteps, pred_ros = plh.calculate_fire_ros(
                y_pred, 
                cell_size=self.cell_size, 
                temporal_resolution=self.temporal_resolution,
                ros_method=self.ros_method,
                interval=self.ros_interval,
                sliding_window=self.ros_sliding_window,
            )
            # logger.debug(true_timesteps.shape, true_ros.shape, pred_timesteps.shape, pred_ros.shape)

            # Find common timesteps between true_timesteps and pred_timesteps
            # common_timesteps = np.intersect1d(true_timesteps, pred_timesteps)

            # # Get indices of common timesteps in each array
            # true_indices = np.where(np.isin(true_timesteps, common_timesteps))[0]
            # pred_indices = np.where(np.isin(pred_timesteps, common_timesteps))[0]

            # # Use only ROS values at common timesteps
            # true_ros = true_ros[true_indices]
            # pred_ros = pred_ros[pred_indices]

            # Fire metrics loss combines ROS and BA differences
            ros_loss_arr = (true_ros - pred_ros)**2

            # print(torch.mean(ba_loss_arr).item(), torch.mean(ros_loss_arr).item())
            if torch.numel(ros_loss_arr) > 0:
                ros_loss = self.fire_metrics_weight * torch.mean(ros_loss_arr)
                loss_final += ros_loss
                loss_components['fire_metrics'] += ros_loss.detach().cpu().item()
        
        return loss_final, loss_components

class PhysicsGuidedLoss:
    def __init__(
        self, 
        use_physics_loss=False, 
        fuel_transport_weight=0, 
        burned_weight=0,
        unburned_weight=0, 
        fire_metrics_weight=0, 
        ros_method=None,
        mse_weight=1.0,
    ):
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        self.use_physics_loss = use_physics_loss
        self.mse_weight = mse_weight
        
        if use_physics_loss:
            logger.info('Physics Loss enabled with weights:')
            logger.info(f'  Fuel Transport: {fuel_transport_weight}')
            logger.info(f'  Burned Area: {burned_weight}')
            logger.info(f'  Unburned Area: {unburned_weight}')
            logger.info(f'  Fire Metrics: {fire_metrics_weight}')
            logger.info(f'  MSE Weight: {mse_weight}')

        self.physics_loss_fn = WildFirePhysicsLoss(
            fuel_transport_weight=fuel_transport_weight, 
            burned_weight=burned_weight, 
            unburned_weight=unburned_weight, 
            fire_metrics_weight=fire_metrics_weight, 
            ros_method=ros_method,
            mse_weight=mse_weight, 
        )
        
    def __call__(self, pred, tar, seq_idx=None):
        if len(pred.shape) > len(tar.shape):
            tar = tar.unsqueeze(1)  # Add channel dimension if missing
        if self.use_physics_loss:
            if seq_idx is not None:
                # Group by sequence and compute loss for each sequence separately
                seq_idx = seq_idx.cpu().numpy()  # Convert to numpy for easier manipulation
                unique_seqs = np.unique(seq_idx)

                # Lists to collect losses and components
                sequence_losses = []
                all_loss_components = {
                    'base_mse': [],
                    'fuel_transport': [],
                    'burned': [],
                    'unburned': [],
                    'fire_metrics': []
                }

                for seq_id in unique_seqs:
                    # Create mask for current sequence
                    seq_mask = (seq_idx == seq_id)

                    # Extract samples for this sequence
                    seq_pred = pred[seq_mask]
                    seq_tar = tar[seq_mask]

                    # Compute loss for this sequence
                    seq_loss, seq_loss_components = self.physics_loss_fn(y_true=seq_tar, y_pred=seq_pred)
                    # logger.debug(seq_loss, seq_loss_components)
                    sequence_losses.append(seq_loss)
                    # Collect components for averaging
                    for key, value in seq_loss_components.items():
                        all_loss_components[key].append(value)
                
                # Average the losses across sequences
                if sequence_losses:
                    total_loss = torch.mean(torch.stack(sequence_losses))
                    # Average loss components
                    avg_loss_components = {k: sum(v)/len(v) if v else 0.0 for k, v in all_loss_components.items()}

                return total_loss, avg_loss_components

            else:
                # Direct batch processing without sequence information
                total_loss, loss_components = self.physics_loss_fn(y_true=tar, y_pred=pred)
                return total_loss, loss_components

        else:
            # Standard MSE loss without physics components
            total_loss = self.mse_weight * self.mse_loss_fn(pred, tar)
            return total_loss, {'base_mse': total_loss.item()}
