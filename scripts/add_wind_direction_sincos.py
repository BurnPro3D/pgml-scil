"""
Script: Add Wind Direction Components to Lite Data

Description:
- This script processes files in the `data_lite/` directory and saves the modified versions to `data_lite_new/`.
- For label files (`-y-` in the filename): copies them as-is.
- For feature files (`-features-` in the filename):
    - Calculates wind direction sine and cosine components.
    - Adds the new components alongside the existing features.
    - Saves the modified data to the destination directory.

Author: Saqib Azim
Date: October 2024
"""

import os
import numpy as np
import shutil
from tqdm import tqdm


# Define source and destination directories
src_dir = "/home/pgmlvol/data/data_lite/"
dst_dir = "/home/pgmlvol/data/data_lite_new/"
os.makedirs(dst_dir, exist_ok=True)  # Ensure the destination directory exists

# Iterate through all files in the source directory
for filename in tqdm(os.listdir(src_dir), desc="Processing files"):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    # Copy label files as-is
    if "-y-" in filename:
        shutil.copy(src_path, dst_path)

    # Process feature files by adding wind direction components
    elif "-features-" in filename:
        data = np.load(src_path)
        # Ensure the data has at least 2 dimensions before indexing
        if data.ndim < 2 or data.shape[-1] < 4:
            print(f"Skipping {filename}: unexpected shape {data.shape}")
            continue

        # Calculate wind direction sin and cos components
        wind_angle = np.deg2rad(data[..., 1] * 100 + 230)
        wind_dir_sin = np.sin(wind_angle)
        wind_dir_cos = np.cos(wind_angle)

        # Stack new components with the original data
        # New order: [original_feature_0, wind_dir_sin, wind_dir_cos, original_feature_2, original_feature_3]
        data = np.stack([
            data[..., 0],         # Original feature 0
            wind_dir_sin,         # Sin component of wind direction
            wind_dir_cos,         # Cos component of wind direction
            data[..., 2],         # Original feature 2
            data[..., 3]          # Original feature 3
        ], axis=-1)

        # Save the modified data
        np.save(dst_path, data)