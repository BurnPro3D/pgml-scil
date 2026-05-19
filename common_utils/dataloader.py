import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

from .dataset import (
    DataPreprocessorLite, 
    SequentialDatasetLite, 
    SequentialDatasetFull, 
    SimpleDataset
)


def create_datasets_lite(config, project_dir, process_rank=0, num_processes=1):
    """Create datasets for lite data"""
    # Load and preprocess data
    dataprocessor_lite = DataPreprocessorLite(project_dir, config.data_dir)
    train_x, train_y, val_x, val_y = dataprocessor_lite.load_data()

    if process_rank == 0:
        logger.info(f"Loaded: {train_x.shape}, {train_y.shape}, {val_x.shape}, {val_y.shape}")
    
    train_x, train_y, val_x, val_y = dataprocessor_lite.preprocess_data(
        train_x=train_x, 
        train_y=train_y, 
        test_x=val_x, 
        test_y=val_y, 
        teacher_forcing=config.teacher_forcing,
        exclude_ignition=config.exclude_ignition, 
        exclude_sourcemap=config.exclude_sourcemap, 
        desired_height=config.image_size, 
        desired_width=config.image_size,
    )
    if process_rank == 0:
        logger.info(f"Modified: {train_x.shape}, {train_y.shape}, {val_x.shape}, {val_y.shape}")

    future = getattr(config, 'future', config.context_len)
    
    # Create datasets
    train_dataset = SequentialDatasetLite(
        inputs=train_x, 
        targets=train_y, 
        context_len=config.context_len, 
        temporal_stride=config.temporal_stride, 
        num_pred_frames=config.num_pred_frames,
        future=future,
        process_rank=process_rank, 
        num_processes=num_processes, 
        create_vis=True
    )
    
    val_dataset = SequentialDatasetLite(
        inputs=val_x, 
        targets=val_y, 
        context_len=config.context_len, 
        temporal_stride=config.temporal_stride, 
        num_pred_frames=config.num_pred_frames, 
        future=future,
        process_rank=process_rank, 
        num_processes=num_processes, 
        create_vis=True
    )
    
    test_dataset = SimpleDataset(
        X=val_x, 
        y=val_y
    )
    
    return train_dataset, val_dataset, test_dataset


def create_datasets_full(config, process_rank=0, num_processes=1):
    """Create datasets for full data"""

    future = getattr(config, 'future', config.context_len)

    train_dataset = SequentialDatasetFull(
        data_dir=os.path.join(config.data_dir, "train"), 
        context_len=config.context_len, 
        temporal_stride=config.temporal_stride, 
        num_pred_frames=config.num_pred_frames, 
        future=future,
        desired_height=config.image_size, 
        desired_width=config.image_size, 
        exclude_ignition=config.exclude_ignition, 
        exclude_sourcemap=config.exclude_sourcemap, 
        teacher_forcing=config.teacher_forcing, 
        process_rank=process_rank, 
        num_processes=num_processes, 
        mode='train', 
        num_files=config.num_train_files, 
        seed=config.seed, 
    )
    
    val_dataset = SequentialDatasetFull(
        data_dir=os.path.join(config.data_dir, "val"), 
        context_len=config.context_len, 
        temporal_stride=config.temporal_stride, 
        num_pred_frames=config.num_pred_frames, 
        future=future,
        desired_height=config.image_size, 
        desired_width=config.image_size, 
        exclude_ignition=config.exclude_ignition, 
        exclude_sourcemap=config.exclude_sourcemap, 
        teacher_forcing=config.teacher_forcing, 
        process_rank=process_rank, 
        num_processes=num_processes, 
        mode='val', 
        num_files=config.num_val_files, 
        seed=config.seed, 
    )
    
    test_dataset = SequentialDatasetFull(
        data_dir=os.path.join(config.data_dir, "test"), 
        context_len=config.context_len, 
        temporal_stride=config.temporal_stride, 
        num_pred_frames=config.num_pred_frames, 
        future=future,
        desired_height=config.image_size, 
        desired_width=config.image_size, 
        exclude_ignition=config.exclude_ignition, 
        exclude_sourcemap=config.exclude_sourcemap, 
        teacher_forcing=config.teacher_forcing, 
        process_rank=process_rank, 
        num_processes=num_processes, 
        mode='test', 
        num_files=config.num_test_files, 
        seed=config.seed, 
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, config, test_batch_size=None):
    """Create DataLoaders from datasets"""
    # NECESSARY shuffle=False in training for PhysicsGuidedLoss to work 
    # as it would provide consecutive samples in a batch necessary for 
    # computing fuel transport and firemetrics loss components
    # shuffle = False if config.physics_loss else True

    # 1. Check if we are running in DDP mode
    is_ddp = dist.is_available() and dist.is_initialized()
    
    # 2. Determine Batch Size
    # In DDP, the batch size in config is usually "per GPU". 
    # If your config.mini_batch_size is the *global* batch size, divide it by world_size.
    # Assuming config.mini_batch_size is ALREADY per-gpu here:
    train_bs = config.mini_batch_size
    val_bs = config.mini_batch_size

    # Use config mini_batch_size unless test_batch_size is overridden (for test loader)
    test_bs = test_batch_size if test_batch_size is not None else config.mini_batch_size

    train_sampler = None
    val_sampler = None
    train_shuffle = False 

    if is_ddp:
        # DistributedSampler handles the shuffling internally via 'shuffle=' arg
        # It also PADS the dataset so all GPUs have equal batches
        train_sampler = DistributedSampler(
            train_dataset, 
            shuffle=train_shuffle, 
            drop_last=False
        )

        val_sampler = DistributedSampler(
            val_dataset, 
            shuffle=False, 
            drop_last=False
        )

   
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.mini_batch_size, 
        shuffle=train_shuffle if train_sampler is None else False,  # shuffle only if not using DistributedSampler
        sampler=train_sampler,
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor, 
        persistent_workers=True,    # workers are not shutdown after each epoch
        pin_memory=True             # for faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.mini_batch_size, 
        shuffle=False,              # not working for IterableDataset
        sampler=val_sampler,
        num_workers=1, 
        prefetch_factor=config.prefetch_factor, 
        persistent_workers=False,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset is not None:

        test_sampler = None
        if is_ddp:
            test_sampler = DistributedSampler(
                test_dataset, 
                shuffle=False, 
                drop_last=False
            )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_bs, # Use the potentially overridden batch size
            shuffle=False,
            sampler=test_sampler,
            num_workers=config.num_workers, 
            prefetch_factor=config.prefetch_factor, 
            persistent_workers=True,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader