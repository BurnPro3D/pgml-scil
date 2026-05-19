import os


def create_run_name(config):
    """Create descriptive run name from key hyperparameters"""
    run_name = f"{config.timestamp}"
    
    # Key identifiers
    data_str = config.which_data.replace("data_", "")
    run_name += f"_{data_str}"
    
    # Physics loss configuration
    if config.physics_loss:
        run_name += f"_phy"
        # Format weights using scientific notation for different scales
        def format_weight(w):
            if w == 0:
                return "0"
            elif w >= 1:
                return f"{w:.0f}" if w == int(w) else f"{w:.1f}".rstrip('0').rstrip('.')
            else:
                return f"{w:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        
        weights = [
            format_weight(config.fuel_transport_weight),
            format_weight(config.burned_weight), 
            format_weight(config.unburned_weight),
            format_weight(config.fire_metrics_weight)
        ]
        weights_str = "_".join(weights)
        run_name += f"_w{weights_str}"
        
        # ROS method abbreviation (only if fire_metrics_weight > 0)
        if config.fire_metrics_weight > 0:
            ros_abbrev = {
                "horizontal_instant": "ins",
                "horizontal_average": "avg", 
                "perimeter_displacement": "dis"
            }.get(config.ros_method, "unk")
            run_name += f"_{ros_abbrev}"
    
    # Teacher forcing
    if config.teacher_forcing:
        run_name += f"_tf"
    
    # Exclusions (only add if True to keep names shorter)
    if config.exclude_ignition:
        run_name += "_noign"
    if config.exclude_sourcemap:
        run_name += "_nosrc"
    
    if config.get("use_fuel_model"):
        run_name += "_fuel"
        if config.get("mixture_type"):
            run_name += f"_{config.get('mixture_type')[:4]}"
    
    STANDARD_MAX_LR = 1e-3
    STANDARD_MIN_LR = 1e-6
    if config.max_lr != STANDARD_MAX_LR or config.min_lr != STANDARD_MIN_LR:
        # Learning rate (fix missing underscore)
        max_lr_str = f"lr{config.max_lr:.0e}".replace("e-0", "e-")
        min_lr_str = f"lr{config.min_lr:.0e}".replace("e-0", "e-")
        run_name += f"_{max_lr_str}_{min_lr_str}"
    
    # Model architecture (only include if different from defaults)
    if config.get("context_len"):
        run_name += f"_ctx{config.context_len}"
    
    if config.get("embed_dim"):
        run_name += f"_emb{config.embed_dim}"
    
    block_val = config.get("depth") or config.get("num_blocks")
    if block_val is not None:
        run_name += f"_blk{block_val}"
    
    # Epochs (only if significantly different from default)
    # if config.max_epochs > 10:  # only add if non-trivial training
    run_name += f"_ep{config.max_epochs}"
    run_name += f"_sd{config.seed}"
    
    return run_name


def setup_experiment_tracking(config):
    """Set up experiment tracking and logging"""

    config.run_id = create_run_name(config)

    if config.logging_framework == 'mlflow':
        import tempfile
        import mlflow
        from common_utils.mlflow_utils import setup_mlflow
        setup_mlflow(
            tracking_uri=config.mlflow_tracking_uri, 
            experiment_name=config.experiment_name, 
            run_id=config.run_id, 
        )
        
        # Save config.yaml as an MLflow artifact
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'config.yaml')
            config.save(path=config_path)
            mlflow.log_artifact(config_path, artifact_path="config")
        
        # Log all parameters from config (can only save params once, so saving at end of training)
        for key, value in config._config.items():
            mlflow.log_param(key, value)
            
    elif config.logging_framework == 'wandb':
        import wandb
        os.environ['WANDB_API_KEY'] = config.wandb_api_key
        wandb.init(
            project=config.experiment_name,
            name=config.run_id,
            config=config._config,
            dir=config.log_dir,
            tags=None,
        )
        
    elif config.logging_framework == 'local':
        # Setup experiment paths with timestamp
        config.exp_dir = os.path.join(config.log_dir, config.run_id)
        config.ckpt_dir = os.path.join(config.exp_dir, "checkpoints")
        
        # Create directories
        for dir_path in [config.log_dir, config.exp_dir, config.ckpt_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Save the config for this training run (at the start of training)
        config.save(path=os.path.join(config.exp_dir, 'config.yaml'))

        # Create a file to log metrics
        with open(os.path.join(config.exp_dir, 'metrics_log.txt'), "w") as f:
            f.write("")  # Clear the file if it exists
