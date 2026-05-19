#!/bin/bash

# This script first installs a specific version of tensorflow-probability
# and then executes a Python training script with specified arguments.

# Exit immediately if any command fails
set -e

# --- 1. Install Python Package ---
echo "Attempting to install tensorflow_probability==0.24.0..."
pip install tensorflow_probability==0.24.0

# Verify installation was successful
if [ $? -eq 0 ]; then
    echo "tensorflow_probability installed successfully."
else
    echo "Error: Failed to install tensorflow_probability. Aborting."
    exit 1
fi

echo "--------------------------------------------------"

# --- 2. Run the Python Training Script ---
echo "Executing the training script..."
python /home/pgmlvol/jjaiswal/pgml/convlstm/src/train/train2.py \
    --experiment_name "pgcl-str1-future5-out3-NOROS" \
    --model_name "cl2" \
    --loss_name "mse" \
    --seed "43" \
    --stride "1" \
    --future "5" \
    --epochs "15" \
    --output_seq_len "3" \
    --lambda_burned "0.01" \
    --lambda_unburned "0.01" \
    --lambda_ros "0.0" \
    --lambda_fuel_transport "0.01" \
    --lambda_mse "1.0" \
    --lambda_consumption "0.01" \
# python /home/pgmlvol/jjaiswal/pgml/convlstm/src/test/test2.py \
#     --run_id "d8b4ff547164406a8108e65b629122c2" \
#     --model_name "cl2" \
#     --loss_name "mse" \
#     --seed "43" \
#     --stride "1" \
#     --future "5" \
#     --epochs "15" \
#     --output_seq_len "3" \
#     --lambda_burned "0.01" \
#     --lambda_unburned "0.01" \
#     --lambda_ros "0.01" \
#     --lambda_fuel_transport "0.01" \
#     --lambda_mse "1.0" \
#     --lambda_consumption "0.01" 
echo "--------------------------------------------------"
echo "Script finished."