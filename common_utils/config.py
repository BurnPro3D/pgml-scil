import os
import yaml
import argparse
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, filepath: str, override_from_env: bool = True, override_from_args: bool = True):
        with open(filepath, 'r') as f:
            # self._config = yaml.safe_load(f)
            # Use a direct assignment to avoid triggering __setattr__
            object.__setattr__(self, '_config', yaml.safe_load(f))
        
        # Then apply overrides
        if override_from_env:
            self._apply_env_overrides()
        if override_from_args:
            self._apply_arg_overrides()
        
        # Filter config parameters to only include relevant ones for the current run.
        # Start with a copy of full config
        filtered_config = self._config.copy()
        
        # Define physics-related parameters to exclude when physics_loss=False
        physics_params = [
            'fuel_transport_weight',
            'burned_weight', 
            'unburned_weight',
            'fire_metrics_weight',
            'ros_method'
        ]

        # Remove physics parameters if physics_loss is False and physics_test is False
        if not self._config.get('physics_loss', False) and not self._config.get('physics_test', False):
            logger.info(f"Removing physics loss hparams from config object")
            for param in physics_params:
                filtered_config.pop(param, None)  # Remove if exists, ignore if not
        
        fuel_model_params = [
            'mixture_type', 
            'max_fuel_density'
        ]

        # Remove fuel model parameters if use_fuel_model is False
        if not self._config.get('use_fuel_model', False):
            logger.info(f"Removing Fuel model params from config object")
            for param in fuel_model_params:
                filtered_config.pop(param, None)  # Remove if exists, ignore if not
        
        # Store filtered config separately or replace original if preferred
        object.__setattr__(self, '_config', filtered_config)
            
        # Apply dynamic path resolution first
        # self._resolve_dynamic_paths()
    
    def get(self, key, default=None):
        """Get config value with optional default (dict-like interface)"""
        return self._config.get(key, default)
    
    def __getattr__(self, name):
        # ensure dot notation works for keys in _config
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Update _config dynamically
        if '_config' in self.__dict__:
            self._config[name] = value
        else:
            # If _config is not yet initialized, fall back to default setattr
            super().__setattr__(name, value)

    def save(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def print_config(self):
        """Print current configuration"""
        logger.info("Current Configuration:")
        logger.info(yaml.dump(self._config, default_flow_style=False))

    def _apply_env_overrides(self):
        """Override config values from environment variables with TRAIN_ prefix"""
        for key in self._config.keys():
            env_key = f"TRAIN_{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # Handle list values (like data_lite_test_sequences)
                if isinstance(self._config[key], list):
                    # Parse comma-separated values
                    self._config[key] = [int(x.strip()) if x.strip().isdigit() else x.strip() 
                                       for x in value.split(',')]
                else:
                    self._config[key] = self._convert_env_value(value, self._config[key])
                logger.info(f"Override from env: {key} = {self._config[key]} (from {env_key})")
    
    def _apply_arg_overrides(self):
        """Override config values from command line arguments"""
        parser = argparse.ArgumentParser(add_help=False)
        
        # Add arguments for each config key
        for key, value in self._config.items():
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                parser.add_argument(arg_name, type=self._str_to_bool, default=None)
            elif isinstance(value, int):
                parser.add_argument(arg_name, type=int, default=None)
            elif isinstance(value, float):
                parser.add_argument(arg_name, type=float, default=None)
            elif isinstance(value, list):
                parser.add_argument(arg_name, nargs='+', default=None, 
                                  help=f"List values for {key} (space-separated)")
            elif isinstance(value, str) or value is None:
                parser.add_argument(arg_name, type=str, default=None)
        
        args, _ = parser.parse_known_args()
        
        # Apply non-None argument values
        for key in self._config.keys():
            arg_key = key.replace('_', '-')
            arg_value = getattr(args, key.replace('-', '_'), None)
            if arg_value is not None:
                # Handle list arguments
                if isinstance(self._config[key], list) and isinstance(arg_value, list):
                    # Try to convert to appropriate types
                    converted_list = []
                    for item in arg_value:
                        if isinstance(self._config[key][0] if self._config[key] else None, int):
                            converted_list.append(int(item))
                        else:
                            converted_list.append(item)
                    self._config[key] = converted_list
                else:
                    self._config[key] = arg_value
                logger.info(f"Override from args: {key} = {self._config[key]}")

    def set_overrides(self, overrides: Dict[str, Any]):
        """Manually set config overrides from a dictionary"""
        for key, value in overrides.items():
            if key in self._config:
                self._config[key] = value
                logger.info(f"Manual override: {key} = {value}")
            else:
                logger.warning(f"Warning: {key} not found in original config, adding as new parameter")
                self._config[key] = value
    
    @staticmethod
    def _str_to_bool(v: str) -> bool:
        """Convert string to boolean"""
        if isinstance(v, bool):
            return v
        if v.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif v.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    