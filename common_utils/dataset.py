import os
import math
import numpy as np
import random
from tqdm import tqdm
import re
from collections import defaultdict
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import VivitImageProcessor
from functools import lru_cache
import logging

# Get the logger for this module
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.feature_list = X
        self.label_list = y
    
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        feature = self.feature_list[idx]
        label = self.label_list[idx]
        return feature, label


class DataPreprocessorLite:
    """ 
    Dataloader for small lite dataset (containing only 50 train, 49 test samples, 
    with each sample upto 50 timesteps)
    """

    def __init__(self, proj_dir, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.proj_dir = proj_dir

    def load_data(self):
        cache_dir = os.path.basename(os.path.dirname(self.data_dir))    # 'data_lite' or 'data_lite_new'

        if os.path.isfile(os.path.join(self.proj_dir, f'data/{cache_dir}/train_x.npy')):
            logger.info("Reading cached lite dataset")
            train_x = np.load(os.path.join(self.proj_dir, f'data/{cache_dir}/train_x.npy'))
            train_y = np.load(os.path.join(self.proj_dir, f'data/{cache_dir}/train_y.npy'))
            test_x = np.load(os.path.join(self.proj_dir, f'data/{cache_dir}/test_x.npy'))
            test_y = np.load(os.path.join(self.proj_dir, f'data/{cache_dir}/test_y.npy'))
        else:
            logger.info("Reading and Caching lite dataset")
            train_features_files = []
            train_labels_files = []
            test_features_files = []
            test_labels_files = []

            # Iterate over files in the folder
            for file in os.listdir(self.data_dir):
                if file.startswith('train-features'):
                    train_features_files.append(file)
                elif file.startswith('train-y'):
                    train_labels_files.append(file)
                elif file.startswith('test-features'):
                    test_features_files.append(file)
                elif file.startswith('test-y'):
                    test_labels_files.append(file)

            # Sort files to ensure the order is correct
            train_features_files.sort()
            train_labels_files.sort()
            test_features_files.sort()
            test_labels_files.sort()

            logger.info(f"Number of train files: {len(train_features_files)}")
            logger.info(f"Number of test files: {len(test_features_files)}")

            # ensure that order of (feature, label) file is matching
            for X_file, y_file in zip(train_features_files, train_labels_files):
                feature_number = int(X_file.split('.')[0].split('-')[-1])
                label_number = int(y_file.split('.')[0].split('-')[-1])
                assert feature_number == label_number, f'Mismatch: {feature_number}, {label_number}'

            for X_file, y_file in zip(test_features_files, test_labels_files):
                feature_number = int(X_file.split('.')[0].split('-')[-1])
                label_number = int(y_file.split('.')[0].split('-')[-1])
                assert feature_number == label_number, f'Mismatch: {feature_number}, {label_number}'

            # Load files into numpy arrays
            train_x = np.array([np.squeeze(np.load(os.path.join(self.data_dir, file))) for file in tqdm(train_features_files)])
            train_y = np.array([np.squeeze(np.load(os.path.join(self.data_dir, file))) for file in tqdm(train_labels_files)])
            test_x = np.array([np.squeeze(np.load(os.path.join(self.data_dir, file))) for file in tqdm(test_features_files)])
            test_y = np.array([np.squeeze(np.load(os.path.join(self.data_dir, file))) for file in tqdm(test_labels_files)])

            os.makedirs(os.path.join(self.proj_dir, f'data/{cache_dir}'), exist_ok=True)

            np.save(os.path.join(self.proj_dir, f'data/{cache_dir}/train_x.npy'), train_x)
            np.save(os.path.join(self.proj_dir, f'data/{cache_dir}/train_y.npy'), train_y)
            np.save(os.path.join(self.proj_dir, f'data/{cache_dir}/test_x.npy'), test_x)
            np.save(os.path.join(self.proj_dir, f'data/{cache_dir}/test_y.npy'), test_y)

        train_y = np.expand_dims(train_y, axis=-1)
        test_y = np.expand_dims(test_y, axis=-1)

        return train_x, train_y, test_x, test_y

    def preprocess_videos(self, videos, config):
        if config.model['vivit']['pretrained_path']:
            logger.info("Loading pretrained image processor")
            image_processor = VivitImageProcessor().from_pretrained(config.model['vivit']['pretrained_path'])
        else:
            logger.info("Using default image processor")
            image_processor = VivitImageProcessor()
            # image_processor = VivitImageProcessor.preprocess(do_rescale=True)

        num_seq = len(videos)
        inputs = []
        for idx in tqdm(range(num_seq), desc="Preprocessing"):
            # preprocessing is done in the same way as the model
            frames = torch.tensor(videos[idx])    # (T, H, W, C)
            # logger.debug(frames.shape, frames.dtype)
            frames = [frames[i] for i in range(len(frames))]
            frames = image_processor.preprocess(
                        videos=frames, 
                        do_resize=True, 
                        do_rescale=False, 
                        offset=False, 
                        do_normalize=False, 
                        input_data_format='channels_first'
                    )    # dict with keys: pixel_values
            frames = np.array(frames['pixel_values'][0])    # (T, C, H, W)
            inputs.append(frames)

        inputs = torch.tensor(np.array(inputs))
        return inputs

    def preprocess_data(
        self, 
        train_x, 
        train_y, 
        test_x, 
        test_y, 
        teacher_forcing=False, 
        exclude_ignition=False, 
        exclude_sourcemap=False, 
        desired_height=-1, 
        desired_width=-1,
    ):
        if exclude_ignition:
            # exclude last channel (ignition pattern)
            train_x = train_x[..., :-1]
            test_x = test_x[..., :-1]
        
        if exclude_sourcemap:
            # exclude first channel (source map fuel density)
            train_x = train_x[..., 1:]
            test_x = test_x[..., 1:]

        if teacher_forcing:
            train_x = np.concatenate([train_x, train_y], axis=-1)    # [50,50,H,W,5]
            test_x = np.concatenate([test_x, test_y], axis=-1)    # [50,50,H,W,5] 

        # (N, T, H, W, C) --> (N, T, C, H, W)
        train_x = np.transpose(train_x, (0,1,4,2,3))
        train_y = np.transpose(train_y, (0,1,4,2,3))
        test_x = np.transpose(test_x, (0,1,4,2,3))
        test_y = np.transpose(test_y, (0,1,4,2,3))

        img_height, img_width = train_x.shape[-2], train_x.shape[-1]

        nrows, ncols = 0, 0    # number of rows and cols to remove
        if desired_height > 0:
            nrows = (img_height - desired_height) // 2
        if desired_width > 0:
            ncols = (img_width - desired_width) // 2

        if nrows > 0 and ncols > 0:
            train_x = train_x[..., nrows:-nrows, ncols:-ncols]    # [50,50,4,H,W]
            train_y = train_y[..., nrows:-nrows, ncols:-ncols]    # [50,50,1,H,W]
            test_x = test_x[..., nrows:-nrows, ncols:-ncols]    # [49,50,4,H,W]
            test_y = test_y[..., nrows:-nrows, ncols:-ncols]    # [49,50,1,H,W]

        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        test_x = torch.tensor(test_x, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.float32)

        return train_x, train_y, test_x, test_y


class SequentialDatasetLite(Dataset):
    def __init__(
        self, 
        inputs, 
        targets, 
        context_len, 
        temporal_stride=1, 
        num_pred_frames=1,
        future=None,     
        process_rank=0, 
        num_processes=1, 
        create_vis=False, 
    ):
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.context_len = context_len
        self.temporal_stride = temporal_stride
        self.num_pred_frames = num_pred_frames

        if future is not None:
            self.future = future
        else:
            self.future = context_len
        

        # Sanity check: 'future' must be at least 'context_len'
        # otherwise the target would start *inside* the input.
        assert self.future >= self.context_len, \
            f"Prediction offset 'future' ({self.future}) must be >= 'context_len' ({self.context_len})"

        self.process_rank = process_rank
        self.num_processes = num_processes
        self.create_vis = create_vis

        num_seq, max_timesteps = inputs.shape[0], inputs.shape[1]
        # The last possible sample must have its *prediction window* end by max_timesteps.
        # Last possible 'win_start' = max_timesteps - future - num_pred_frames
        self.samples_per_seq = max(0, (max_timesteps - self.future - self.num_pred_frames + self.temporal_stride) // self.temporal_stride)
        total_samples = num_seq * self.samples_per_seq

        # Distribute samples across processes
        self.samples_per_process = total_samples // num_processes
        if total_samples % num_processes > process_rank:
            self.samples_per_process += 1

        # Calculate global offset for this process's portion of the dataset
        # This ensures each process gets a unique, non-overlapping portion of the dataset
        self.global_offset = self.samples_per_process * process_rank + min(process_rank, total_samples % num_processes)

    def __len__(self):
        """Return the number of samples assigned to this process."""
        return self.samples_per_process

    def __getitem__(self, idx):
        global_idx = self.global_offset + idx

        seq_idx = (global_idx // self.samples_per_seq)
        win_start = (global_idx % self.samples_per_seq) * self.temporal_stride

        input_win_end = win_start + self.context_len
        pred_win_start = win_start + self.future
        pred_win_end = pred_win_start + self.num_pred_frames


        # win_end = win_start + self.context_len
        # logger.debug(seq_idx, "(", win_start, ":", win_end, ")")

        # construct single sample in a batch
        x = self.inputs[seq_idx, win_start:input_win_end]
        y = self.targets[seq_idx, pred_win_start:pred_win_end]

        return x, y, seq_idx


def parse_filename(filename):
    """ Parses filename and extract categories """
    match = re.search(r"ws(\d+)_wd(\d+)_sm(\d+)_(\w+)\.npz", filename)
    if match:
        wind_speed = int(match.group(1))
        wind_direction = int(match.group(2))
        # surface_moisture = int(match.group(3))
        ignition_pattern = match.group(4)
        return (wind_speed, wind_direction, ignition_pattern)
    return None


def stratified_sampling(filenames, num_files):
    # Parse filenames and group by categories
    category_groups = defaultdict(list)
    for filename in filenames:
        categories = parse_filename(filename)
        if categories:
            category_groups[categories].append(filename)

    # Calculate the number of samples per category
    total_categories = len(category_groups)
    samples_per_category = max(1, num_files // total_categories)

    # Select files using stratified sampling
    selected_filenames = []
    for category, files in category_groups.items():
        # Randomly sample from each category group
        selected_filenames.extend(random.sample(files, min(samples_per_category, len(files))))

    # Shuffle the selected filenames for randomness
    random.shuffle(selected_filenames)

    return selected_filenames[:num_files]  # Limit to num_files if over-sampled




class SequentialDatasetFull(Dataset):
    """ 
    Map-Style Dataset for Full Dataset.
    Compatible with DistributedSampler (Solves DDP Deadlock).
    """
    
    def __init__(
        self, 
        data_dir, 
        context_len, 
        temporal_stride=1, 
        num_pred_frames=1, 
        future=8, # Added future param
        desired_height=-1, 
        desired_width=-1, 
        exclude_ignition=False, 
        exclude_sourcemap=False, 
        teacher_forcing=False, 
        process_rank=0, 
        num_processes=1, 
        mode='train', 
        num_files=100, 
        create_vis=False, 
        vis_filenames=None, 
        seed=43, 
    ):
        super().__init__()
        self.data_dir = data_dir
        self.exclude_ignition = exclude_ignition
        self.exclude_sourcemap = exclude_sourcemap
        self.teacher_forcing = teacher_forcing
        self.context_len = context_len
        self.temporal_stride = temporal_stride
        self.num_pred_frames = num_pred_frames
        self.future = future if future is not None else context_len
        self.create_vis = create_vis
        # --- 1. File Selection ---
        if create_vis and vis_filenames is not None:
            self.filenames = vis_filenames
        else:
            all_filenames = sorted(os.listdir(data_dir))
            if not self.exclude_ignition:
                all_filenames = [f for f in all_filenames if not f.endswith('StripSouthwards.npz')]
            
            if mode == 'test':
                self.filenames = all_filenames
            else:
                random.seed(seed)
                # Simple random selection (Stratified logic can be added here if needed)
                self.filenames = random.sample(all_filenames, min(len(all_filenames), num_files))
                self.filenames.sort()

        self.filepaths = [os.path.join(data_dir, f) for f in self.filenames]
        
        # --- 2. Index Mapping (The "Map" in Map-Style) ---
        
        # Checking first file for dimensions
        sample_input = np.load(self.filepaths[0])
        # shape: (Timesteps, Channels, H, W) -> e.g. (601, 4, 296, 296)
        full_shape = sample_input['fuel_density'].shape
        self.file_timesteps = full_shape[0]
        img_h, img_w = full_shape[-2], full_shape[-1]

        # 2. Correctly Calculate Margins
        # Logic: (Actual - Desired) / 2
        # Example: (300 - 296) / 2 = 2 pixels to crop
        self.nrows = (img_h - desired_height) // 2 if desired_height > 0 else 0
        self.ncols = (img_w - desired_width) // 2 if desired_width > 0 else 0
        
        # Calculate valid start indices per file
        # Formula: total_time - future_offset - prediction_window
        last_valid_start = self.file_timesteps - self.future - self.num_pred_frames + 1
        self.valid_starts = list(range(0, last_valid_start, self.temporal_stride))
        
        self.samples_per_file = len(self.valid_starts)
        self.total_samples = len(self.filepaths) * self.samples_per_file

    def __len__(self):
        return self.total_samples

    @lru_cache(maxsize=1) 
    def _load_file_cached(self, filepath):
        """ Cached file loader to avoid disk thrashing """
        sample = np.load(filepath)
        # ... (Your Exact Preprocessing Logic) ...
        # Simplified for brevity, insert your exact logic here:
        shp = sample['fuel_density'].shape
        wind_speed = np.full(shp, sample['wind_speed'])
        wind_dir_sin = np.full(shp, sample['wind_direction_sin'])
        wind_dir_cos = np.full(shp, sample['wind_direction_cos'])
        inputs = np.concatenate([sample['source_map'], wind_dir_sin, wind_dir_cos, wind_speed, sample['ignition_pattern']], axis=1)
        targets = sample['fuel_density']

        if self.exclude_ignition: inputs = inputs[:, :-1]
        if self.exclude_sourcemap: inputs = inputs[:, 1:]
        if self.teacher_forcing: inputs = np.concatenate([inputs, sample['fuel_density']], axis=1)
        
        if self.nrows > 0:
            inputs = inputs[..., self.nrows:-self.nrows, self.ncols:-self.ncols]
            targets = targets[..., self.nrows:-self.nrows, self.ncols:-self.ncols]
            
        return inputs, targets

    def __getitem__(self, idx):
        """ 
        Global Index -> (File Index, Time Index) 
        """
        # 1. Map global index to specific file and specific time
        file_idx = idx // self.samples_per_file
        time_idx_local = idx % self.samples_per_file
        
        curr_pos = self.valid_starts[time_idx_local]
        filepath = self.filepaths[file_idx]
        filename = self.filenames[file_idx].split('.')[0]

        # 2. Load Data (Cached)
        inputs, targets = self._load_file_cached(filepath)

        # 3. Slice
        x = inputs[curr_pos : curr_pos + self.context_len]
        y = targets[curr_pos + self.future : curr_pos + self.future + self.num_pred_frames]

        # 4. To Tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.create_vis:
            return x, y, file_idx, filename
        else:
            return x, y, file_idx


class FullSequenceDataset(Dataset):
    """ 
    Dataset for loading FULL sequences one by one for testing. 
    Does not slice into windows. Used for run_seq2seq_test on large datasets.
    """
    def __init__(
        self, 
        config,
        data_dir,
    ):
        self.data_dir = data_dir
        self.config = config
        
        # Filter for valid files (same logic as SequentialDatasetFull)
        all_filenames = sorted(os.listdir(data_dir))
        if not config.exclude_ignition:
            all_filenames = [f for f in all_filenames if not f.endswith('StripSouthwards.npz')]
        
        # Create full paths
        self.filepaths = [os.path.join(data_dir, f) for f in all_filenames]
        
        # Pre-calculate crop dimensions based on the first file
        self.nrows, self.ncols = 0, 0
        if len(self.filepaths) > 0:
            # Load metadata from first file to determine cropping
            sample = np.load(self.filepaths[0])
            # shape is (T, 1, H, W)
            _, _, img_height, img_width = sample['fuel_density'].shape
            
            if config.image_size > 0:
                self.nrows = (img_height - config.image_size) // 2
                self.ncols = (img_width - config.image_size) // 2

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        sample = np.load(filepath)
        shp = sample['fuel_density'].shape
        
        wind_speed = np.full(shp, sample['wind_speed'])
        wind_dir_sin = np.full(shp, sample['wind_direction_sin'])
        wind_dir_cos = np.full(shp, sample['wind_direction_cos'])
        # wind_dir = np.full(shp, math.sqrt(sample['wind_direction_sin']**2 + sample['wind_direction_cos']**2))
        
        inputs = np.concatenate([sample['source_map'], wind_dir_sin, wind_dir_cos, wind_speed, sample['ignition_pattern']], axis=1)
        targets = sample['fuel_density']

        if self.config.exclude_ignition: inputs = inputs[:, :-1]
        if self.config.exclude_sourcemap: inputs = inputs[:, 1:]
        if self.config.teacher_forcing: inputs = np.concatenate([inputs, sample['fuel_density']], axis=1)
        
        if self.nrows > 0 and self.ncols > 0:
            inputs = inputs[..., self.nrows:-self.nrows, self.ncols:-self.ncols]
            targets = targets[..., self.nrows:-self.nrows, self.ncols:-self.ncols]

        # Return (T, C, H, W) tensors. No T transpose needed as load_file/numpy logic 
        # usually keeps T first (T, 1, H, W) in source files, but preprocessor did 
        # (N, T, H, W, C) -> (N, T, C, H, W). 
        # Here inputs is (T, C, H, W) directly from concatenation if source is (T, 1, H, W).
        # np.load gives (601, 1, 300, 300).
        # axis=1 concat preserves (T, C, H, W).
        
        x = torch.from_numpy(inputs).float()
        y = torch.from_numpy(targets).float()
        
        return x, y