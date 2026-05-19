import os
import logging
import torch
import torch.distributed as dist

# Get a logger for this module
logger = logging.getLogger(__name__)


def setup_distributed_training():
    """
    Setup distributed data parallel (DDP) training environment
    'torchrun' command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    RANK and LOCAL_RANK same for (single node, multi-GPU) settings, may differ for (multinode, 
    multi GPU) settings. 
    """
    ddp = int(os.environ.get('RANK', -1)) != -1     # if this is a ddp run or not
    if not ddp:
        # Not using DDP
        return {
            'ddp': False,
            'ddp_rank': 0,
            'ddp_local_rank': 0,
            'ddp_world_size': 1,
            'master_process': True,
            'device': 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        }
    
    # DDP setup
    assert torch.cuda.is_available(), "Use of DDP requires CUDA, but CUDA not available"
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # master process (arbitrarily set to 0) will do printing, logging, checkpointing, etc.
    master_process = ddp_rank == 0
    logger.info(f"Distributed Training | GPU global rank: {ddp_rank}/{ddp_world_size-1}, local GPU: {ddp_local_rank}, {ddp_world_size} processes")
    
    return {
        'ddp': True,
        'ddp_rank': ddp_rank,
        'ddp_local_rank': ddp_local_rank,
        'ddp_world_size': ddp_world_size,
        'master_process': master_process,
        'device': device
    }
