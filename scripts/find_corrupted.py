import os
import yaml

# Your MLflow tracking URI
mlflow_root = "/home/pgmlvol/mlflow2"

print(f"Scanning {mlflow_root} for corrupt runs...")

# Walk through the directory structure
for root, dirs, files in os.walk(mlflow_root):
    if "meta.yaml" in files:
        meta_path = os.path.join(root, "meta.yaml")
        
        try:
            with open(meta_path, "r") as f:
                content = yaml.safe_load(f)
                
            # Check if the content is None (empty file) or not a dictionary
            if content is None:
                print(f"[CORRUPT - EMPTY]: {meta_path}")
            elif not isinstance(content, dict):
                print(f"[CORRUPT - INVALID]: {meta_path}")
                
        except Exception as e:
            print(f"[READ ERROR]: {meta_path} - {e}")

print("Scan complete.")