import torch
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import your model classes here ---
# Adjust these imports based on your exact file structure
from ViViT.src.model import DensityPrediction 
from FourCastNet.networks.afnonet_wf import AFNONet_Seq2Seq 
from convlstm_new.model import PhysicsEnhancedConvLSTM 

def benchmark_model(model_name, model, dummy_input, frames_to_predict=50, frames_per_pass=4, repetitions=10, device='cuda'):
    print(f"\nBenchmarking {model_name}...")
    model = model.to(device)
    model.eval()  # Sets self.training = False
    dummy_input = dummy_input.to(device)
    
    # Calculate how many forward passes are needed to reach 50 timesteps
    passes_needed = math.ceil(frames_to_predict / frames_per_pass)
    
    # 1. GPU Warm-up
    # PyTorch allocates memory dynamically. The first few passes are always 
    # artificially slow. We must warm up the GPU first.
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # 2. Benchmarking Loop
    print(f"Running {repetitions} repetitions (Predicting {frames_to_predict} frames per rep)...")
    times = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            
            # Start timer for this full 50-step sequence
            start_event.record()
            
            # Simulate autoregressive/sliding window forward passes
            current_input = dummy_input.clone()
            for _ in range(passes_needed):
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    out = model(current_input)
                
                # Note: PhysicsEnhancedConvLSTM returns a tuple only when self.training=True.
                # Since we called model.eval(), it safely returns just the final_pred tensor.
                if isinstance(out, tuple):
                    out = out[0]
                    
                # In a real autoregressive loop, you would concat 'out' back into 'current_input'
                # For pure compute-time benchmarking, this forward pass is sufficient.
            
            # Stop timer
            end_event.record()
            torch.cuda.synchronize() # Wait for GPU to finish all operations
            
            # Calculate time in seconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_time_ms / 1000.0) # Convert ms to seconds

    # 3. Calculate Metrics
    min_time = np.min(times)
    mean_time = np.mean(times)
    max_time = np.max(times)
    
    return {
        "Model": model_name,
        "Minimum (s)": round(min_time, 2),
        "Mean (s)": round(mean_time, 2),
        "Maximum (s)": round(max_time, 2)
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not detected. Benchmarking on CPU will not reflect real-world metrics.")

    # Define standard input dimensions based on your configs
    B = 1         # Batch size of 1 for sequential inference testing
    C = 4         # Input channels
    T = 6         # Context length
    H = 296       # Image height
    W = 296       # Image width
    
    frames_per_pass = 3 # num_pred_frames from your config
    dummy_input = torch.randn(B, T, C, H, W, dtype=torch.float32)

    print("Initializing Models...")
    
    # 1. Initialize ConvLSTM (PhysicsEnhancedConvLSTM)
    convlstm = PhysicsEnhancedConvLSTM(
        img_size=(H, W), 
        in_chans=C, 
        out_chans=1
    ) 
    
    # 2. Initialize AFNONet (AFNONet_Seq2Seq)
    afnonet = AFNONet_Seq2Seq(
        img_size=(H, W), patch_size=(8, 8), in_chans=C, out_chans=1,
        context_len=T, temporal_patch_size=4, embed_dim=128, depth=4, num_blocks=4
    )
    
    # 3. Initialize ViViT (DensityPrediction)
    class MockConfig:
        image_size = H
        in_channels = C
        out_channels = 1
        context_len = T
        future = 8
        num_pred_frames = frames_per_pass
        tubelet_size = [4, 8, 8]
        embed_dim = 128
        hidden_dim = 256
        num_blocks = 4
        num_attn_heads = 4
        dropout = 0.1
        pretrained_path = "" # Added to prevent attribute error in DensityPrediction
    
    vivit = DensityPrediction(MockConfig(), device=device)

    # Run Benchmarks
    results = []
    results.append(benchmark_model("ConvLSTM", convlstm, dummy_input, device=device))
    results.append(benchmark_model("AFNONet", afnonet, dummy_input, device=device))
    results.append(benchmark_model("ViViT", vivit, dummy_input, device=device))
    
    # Create and print the final table
    df = pd.DataFrame(results)
    
    # Format table to match the layout of the paper
    df_transposed = df.set_index("Model").T
    
    print("\n\n" + "="*60)
    print("Table 4: Model inference time summary statistics")
    print("for predicting test sequence with 50 time steps")
    print("(in seconds) averaged over 10 repetitions.")
    print("="*60)
    print(df_transposed.to_string())
    print("="*60)

if __name__ == "__main__":
    main()