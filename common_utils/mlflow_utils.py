import os
import mlflow
import torch
import tempfile
from pathlib import Path
import yaml
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri, experiment_name, run_id):
    # setup_logger(self.config)
    mlflow.set_tracking_uri(tracking_uri)

    # Ensure the experiment exists (folders on the side)
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)
        logger.info("MLFlow experiment created")
    else:
        logger.info("MLFlow experiment already exists")
    mlflow.set_experiment(experiment_name)

    # Start a new run (experiment runs inside each folder)
    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_id)
        logger.info("MLFlow run started")

    # configure MLflow autologger
    mlflow.pytorch.autolog(
        log_every_n_epoch=1,
        log_models=True,
        log_datasets=False,  # Set to True only if you want to log dataset info
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        registered_model_name=None,
    )


def log_model_to_mlflow(model, artifact_path, create_signature=True, input_example=None, code_paths=None, model_name=None):
    """Helper method to log a model to MLflow with optional signature"""

    # Temporary CPU copy for logger
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    
    try:
        # Determine whether to create signature
        if create_signature and input_example is not None:
            try:
                # Run inference for signature
                with torch.no_grad():
                    predictions = model_cpu(input_example.unsqueeze(0))
                
                # Create signature: helps MLflow understand the input/output format and 
                # makes loading the model for inference much easier later
                input_sample = input_example.unsqueeze(0).numpy()
                output_sample = predictions.numpy()
                signature = mlflow.models.infer_signature(input_sample, output_sample)
                
                # Save ONE comprehensive model package
                mlflow.pytorch.log_model(
                    model_cpu,
                    artifact_path=artifact_path,    # Standard path for loading
                    signature=signature,
                    input_example=input_sample,
                    code_paths=code_paths,    # Include model definitions
                    # extra_files=["config.yaml"],
                    pip_requirements=["torch>=1.8.0", "pyyaml>=5.1"],
                    registered_model_name=model_name, 
                )
                logger.info(f"Logged model to {artifact_path} with signature")

            except Exception as e:
                logger.error(f"Error creating signature: {e}, logger model without signature")
                # Fall back to logger without signature
                mlflow.pytorch.log_model(model_cpu, artifact_path=artifact_path, registered_model_name=model_name)
        else:
            # Log without signature
            mlflow.pytorch.log_model(model_cpu, artifact_path=artifact_path, registered_model_name=model_name)
    finally:
        # Move model back to original device
        model = model_cpu.to(original_device)


def load_from_mlflow(
    run_id: str, 
    tracking_uri = "/home/pgmlvol/mlflow/", 
    output_dir: str = "./loaded_artifacts", 
    model_initializer=None
):
    """
    Load model and config from MLflow run using the saved model artifact
    rather than reconstructing the model architecture from code.
    
    Args:
        run_id: MLflow run ID (just the UUID, e.g. "0100ac4874b74e2480f9b5f196b4c95f")
        tracking_uri: MLflow tracking URI
        output_dir: Local directory to save downloaded artifacts
        model_initializer: Optional function that takes config and returns initialized model
                          Used as fallback when direct model loading fails
    Returns:
        Tuple of (loaded_model, config_dict)
    """
    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Set MLflow tracking URI: {tracking_uri}")

    # Verify run exists
    try:
        run = mlflow.get_run(run_id)
        logger.info(f"Loading from run: {run.info.run_name} (ID: {run_id})")
    except Exception as e:
        raise ValueError(f"Run {run_id} not found at tracking URI {mlflow.get_tracking_uri()}") from e
    
    print('loading artifacts')
    # Load config.yaml
    config_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/config/config.yaml",
        dst_path=str(output_dir)
    )
    print('config_path', config_path)
    logger.info(f"Config path: {config_path}")

    # Load config using your Config class if available, otherwise use plain dict
    try:
        from common_utils.config import Config
        config = Config(config_path)
    except ImportError:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.warning("Config loaded as plain dictionary (Config class not found)")
    
    
    # Load the complete model (architecture + weights) from MLflow
    # Try different approaches in sequence
    model = None
    
    # Approach 1: Try loading as PyTorch model directly
    try:
        model_path = f"runs:/{run_id}/best_model"
        model = mlflow.pytorch.load_model(model_path)
        logger.info(f"Successfully loaded PyTorch model from {model_path}")
    except Exception as e:
        logger.error(f"Could not load as PyTorch model: {e}")
        
        # Approach 2: Try as a pyfunc model
        try:
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/best_model")
            logger.info(f"Successfully loaded as generic MLflow model")
        except Exception as e2:
            logger.error(f"Could not load as generic model either: {e2}")
            
            # Approach 3: Try to load raw checkpoint
            try:
                logger.info("Attempting to load raw checkpoint...")
                ckpt_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/checkpoints/best_model.pth",
                    dst_path=str(output_dir)
                )
                checkpoint = torch.load(ckpt_path)
                
                # Use the provided model initializer if available
                if model_initializer:
                    logger.info("Using provided model initializer function")
                    model = model_initializer(config)
                    model.load_state_dict(checkpoint['state_dict'])
                    logger.info("Successfully loaded model from raw checkpoint using initializer")
                else:
                    logger.warning("WARNING: No model initializer provided. Cannot create model from checkpoint only.")
                    logger.warning("Returning config and checkpoint dictionary instead")
                    return checkpoint, config
            except Exception as e3:
                logger.error(f"Failed to load checkpoint: {e3}")
                logger.error("Returning config only")
                return None, config
    
    # Move model to appropriate device if available
    if model:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model loaded and moved to {device}")
    
    logger.info(f"Successfully loaded artifacts from run {run_id}")
    return model, config