import torch


def generate_predictions(model, val_loader, device='cuda', teacher_forcing=False):
    """
    Generate predictions for sequences in the validation loader with option for teacher forcing.
    
    Args:
        model: Loaded model from MLflow
        val_loader: DataLoader containing validation sequences
        device: Device to run inference on ('cuda' or 'cpu')
        use_teacher_forcing: Whether to use teacher forcing during inference
        fuel_density_channel: Channel index where fuel density is located in the input when using teacher forcing
        
    Returns:
        Dictionary containing predictions, targets, and inputs for each sequence
    """
    model.eval()
    results = {
        'predictions': {},  # Will store predictions for each sequence
        'targets': {},      # Will store ground truth for each sequence
        # 'inputs': {}        # Will store input data for each sequence
    }
    
    with torch.no_grad():
        for batch_idx, (inp, tar, seq_idx) in enumerate(val_loader):
            inp, tar = inp.squeeze(1), tar.squeeze(1)
            inp = inp.to(device, dtype=torch.float, non_blocking=True)
            tar = tar.to(device, dtype=torch.float, non_blocking=True)

            # Process each sequence separately
            for seq_id in torch.unique(seq_idx):
                # Get samples for this sequence
                seq_mask = (seq_idx == seq_id)
                seq_inputs = inp[seq_mask].to(device)
                seq_targets = tar[seq_mask].to(device)
                
                # Initialize storage for this sequence if not already present
                seq_id_str = seq_id.item()
                if seq_id_str not in results['predictions']:
                    results['predictions'][seq_id_str] = []
                    results['targets'][seq_id_str] = []
                    # results['inputs'][seq_id_str] = []
                
                # Store the original inputs and targets
                # results['inputs'][seq_id_str].append(seq_inputs.cpu())
                results['targets'][seq_id_str].append(seq_targets.cpu())
                
                # Generate predictions
                if teacher_forcing:
                    # Autoregressive prediction with teacher forcing
                    seq_preds = []
                    prev_fuel_density = None
                    
                    for i in range(len(seq_inputs)):
                        current_inp = seq_inputs[i:i+1].clone()  # Keep batch dimension, create a copy
                        
                        # For steps after the first, replace ground truth fuel density with previous prediction
                        if i > 0 and prev_fuel_density is not None:
                            current_inp[0, -1, :, :] = prev_fuel_density[0, 0, :, :]
                        
                        # Predict next step
                        pred = model(current_inp)
                        seq_preds.append(pred)
                        prev_fuel_density = pred  # Save for next iteration
                    
                    # Combine all predictions for this sequence
                    seq_preds = torch.cat(seq_preds, dim=0)
                    results['predictions'][seq_id_str].append(seq_preds.cpu())
                
                else:
                    # Standard inference without teacher forcing
                    seq_preds = model(seq_inputs)
                    results['predictions'][seq_id_str].append(seq_preds.cpu())

    # Concatenate results for each sequence
    for seq_id in results['predictions'].keys():
        results['predictions'][seq_id] = torch.cat(results['predictions'][seq_id], dim=0)
        results['targets'][seq_id] = torch.cat(results['targets'][seq_id], dim=0)
        # results['inputs'][seq_id] = torch.cat(results['inputs'][seq_id], dim=0)
    
    return results