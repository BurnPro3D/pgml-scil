import torch
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import your model classes here ---
from ViViT.src.model import DensityPrediction 
from FourCastNet.networks.afnonet_wf import AFNONet_Seq2Seq 
from convlstm_new.model import PhysicsEnhancedConvLSTM 

def benchmark_model(model_name, model, dummy_input, frames_to_predict=50, frames_per_pass=3, stride=1, repetitions=490, device='cuda'):
    print(f"\nBenchmarking {model_name} (Autoregressive, Stride={stride})...")
    model = model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)
    
    B, T_in, C, H, W = dummy_input.shape
    passes_needed = math.ceil(frames_to_predict / stride)
    
    # 1. GPU Warm-up
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # 2. Benchmarking Loop
    print(f"Running {repetitions} repetitions (Predicting {frames_to_predict} total frames via {passes_needed} forward passes)...")
    times = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Pre-allocate a dummy weather/topography feature tensor.
    # In a real run, this is fetched from the ground truth. Here, we use it to 
    # perfectly mimic the memory bandwidth/overhead of merging the prediction 
    # back into the 5-channel input tensor.
    dummy_next_features = torch.randn(B, stride, C, H, W, device=device, dtype=torch.float32)

    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            
            start_event.record()
            current_input = dummy_input.clone()
            
            for _ in range(passes_needed):
                # Using float16 for Titan RTX compatibility 
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    out = model(current_input)
                
                if isinstance(out, tuple):
                    out = out[0]
                    
                # A. Take the first 'stride' amount of frames from the predicted sequence
                # out shape: (B, frames_per_pass, C_out=1, H, W)
                pred_stride = out[:, :stride, ...]
                
                # B. Inject the predicted fuel density (C=1) into the full feature tensor (C=5).
                # Assuming fuel density is the last channel (index -1)
                dummy_next_features[:, :, -1:, ...] = pred_stride.float() 
                
                # C. Roll the input sequence backward and append the new frames
                current_input = torch.roll(current_input, shifts=-stride, dims=1)
                current_input[:, -stride:, ...] = dummy_next_features
            
            end_event.record()
            torch.cuda.synchronize() 
            
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_time_ms / 1000.0) 

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

    # Standard input dimensions
    B = 1         
    C = 4         
    T = 6         
    H = 296       
    W = 296       
    
    # Evaluation settings specified by user
    frames_per_pass = 3 
    stride = 1          
    frames_to_predict = 50

    dummy_input = torch.randn(B, T, C, H, W, dtype=torch.float32)

    print("Initializing Models...")
    
    # 1. Initialize ConvLSTM
    convlstm = PhysicsEnhancedConvLSTM(
        img_size=(H, W), 
        in_chans=C, 
        out_chans=1
    ) 
    
    # 2. Initialize AFNONet
    afnonet = AFNONet_Seq2Seq(
        img_size=(H, W), patch_size=(8, 8), in_chans=C, out_chans=1,
        context_len=T, temporal_patch_size=frames_per_pass, embed_dim=128, depth=4, num_blocks=4
    )
    
    # 3. Initialize ViViT 
    class MockConfig:
        image_size = H
        in_channels = C
        out_channels = 1
        context_len = T
        future = 8
        num_pred_frames = frames_per_pass
        tubelet_size = [3, 4, 4]
        embed_dim = 256
        hidden_dim = 512
        num_blocks = 4
        num_attn_heads = 4
        dropout = 0.1
        pretrained_path = "" 
    
    vivit = DensityPrediction(MockConfig(), device=device)

    # Run Benchmarks
    results = []
    results.append(benchmark_model("ConvLSTM", convlstm, dummy_input, frames_to_predict, frames_per_pass, stride, device=device))
    results.append(benchmark_model("AFNONet", afnonet, dummy_input, frames_to_predict, frames_per_pass, stride, device=device))
    results.append(benchmark_model("ViViT", vivit, dummy_input, frames_to_predict, frames_per_pass, stride, device=device))
    
    df = pd.DataFrame(results)
    df_transposed = df.set_index("Model").T
    
    print("\n\n" + "="*60)
    print("Table 4: Model inference time summary statistics")
    print(f"for predicting test sequence with {frames_to_predict} time steps")
    print("(in seconds) averaged over 10 repetitions.")
    print("="*60)
    print(df_transposed.to_string())
    print("="*60)

if __name__ == "__main__":
    main()