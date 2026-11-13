"""Builds the PGML model and its loss for FIRETEC data."""

from common_utils.config import Config
from common_utils.loss import PhysicsGuidedLoss
from FourCastNet.networks.afnonet_wf import AFNONet_Seq2Seq
from firetec_pgml.features import INPUT_CHANNELS, RunGeometry


def build_model(config: Config, geometry: RunGeometry) -> AFNONet_Seq2Seq:
    """The repository's AFNO sequence-to-sequence model, sized for the runs' grid."""
    _check_model_fits(config, geometry.grid_shape)
    return AFNONet_Seq2Seq(
        img_size=geometry.grid_shape,
        patch_size=(config.patch_size, config.patch_size),
        in_chans=len(INPUT_CHANNELS),
        out_chans=config.out_channels,
        context_len=config.context_len,
        temporal_patch_size=config.temporal_patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_blocks=config.num_blocks,
    )


def build_loss(config: Config, geometry: RunGeometry) -> PhysicsGuidedLoss:
    """The repository's physics-guided loss, told the runs' cell size and frame interval."""
    loss = PhysicsGuidedLoss(
        use_physics_loss=config.physics_loss,
        fuel_transport_weight=config.get("fuel_transport_weight", 0),
        burned_weight=config.get("burned_weight", 0),
        unburned_weight=config.get("unburned_weight", 0),
        fire_metrics_weight=config.get("fire_metrics_weight", 0),
        ros_method=config.get("ros_method"),
        mse_weight=config.mse_weight,
    )
    # The loss assumes QUIC-Fire's 2 m cells and 1 s frames unless told otherwise.
    loss.physics_loss_fn.cell_size = geometry.cell_size_m
    loss.physics_loss_fn.temporal_resolution = geometry.time_step_s
    return loss


def _check_model_fits(config: Config, grid_shape: tuple[int, int]) -> None:
    if any(cells % config.patch_size for cells in grid_shape):
        raise ValueError(f"patch_size {config.patch_size} does not divide the grid {grid_shape}")
    if config.context_len % config.temporal_patch_size:
        raise ValueError(
            f"temporal_patch_size {config.temporal_patch_size} does not divide "
            f"context_len {config.context_len}"
        )
    if config.num_pred_frames > config.context_len:
        raise ValueError(
            f"the model returns context_len = {config.context_len} frames, fewer than "
            f"num_pred_frames = {config.num_pred_frames}"
        )
