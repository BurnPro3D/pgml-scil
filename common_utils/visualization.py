import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import cmocean.cm as cmo
import logging

logger = logging.getLogger(__name__)


def create_animation(
    true_data, 
    fig_title=None, 
    fps=30,
    cmap=cmo.thermal_r, 
    title='Evolution',
    save_path=None, 
    dpi=100,  # Lower DPI for faster rendering
    jpeg_quality=80  # JPEG quality for MP4 compression
):
    """
    Create a single-panel animation
    Args:
        true_data: Ground truth data [timesteps, height, width]
        interval: Time between frames in milliseconds
        fps: Frames per second for saved video
        cmap: Colormap for true/pred visualization, choices: ['YlOrRd', 'viridis', cmo.thermal_r]
        title: Title for the animation
        save_path: Path to save the animation
    """
    if isinstance(true_data, torch.Tensor):
        true_data = true_data.float().detach().cpu().numpy()
    # print("True data shape:", true_data.shape)

    # Create figure with three subplots
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=dpi)
    plt.close()
    
    # Value ranges
    vmin = np.min(true_data)
    vmax = np.max(true_data)

    # Initialize plots with minimal decorations
    ax.set_xticks([])
    ax.set_yticks([])

    # Initialize plots
    im1 = ax.imshow(true_data[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)

    # Add colorbars
    fig.colorbar(im1, ax=ax)

    # Set titles
    ax.set_title('True')
    if fig_title is not None:
        fig.suptitle(f'{title} | {fig_title} \nFrame: 1/{len(true_data)}')
    else:
        fig.suptitle(f'{title}\nFrame: 1/{len(true_data)}')
    
    def update(frame):
        im1.set_array(true_data[frame])
        if fig_title is not None:
            fig.suptitle(f'{title} | {fig_title}\nFrame: {frame+1}/{len(true_data)}')
        else:
            fig.suptitle(f'{title}\nFrame: {frame+1}/{len(true_data)}')
        return [im1]
    
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(true_data),
        interval=1000/fps,
        blit=True, 
        cache_frame_data=False  # Reduce memory usage
    )

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            # For MP4, use ffmpeg with optimized settings
            writer = animation.FFMpegWriter(
                fps=fps,
                codec='h264',
                bitrate=-1,  # Let ffmpeg determine bitrate
                extra_args=[
                    '-preset', 'ultrafast',
                    '-crf', '30',  # Higher CRF = more compression
                    '-pix_fmt', 'yuv420p',  # Required for compatibility
                    '-tune', 'animation',  # Optimize for animation content
                    '-threads', 'auto'  # Use all available CPU cores
                ]
            )
            anim.save(save_path, writer=writer)
    return anim


class ClippingNormalize(Normalize):
    def __init__(self, vmin, vmax, clip_below=None, clip_above=None):
        self.clip_below = clip_below
        self.clip_above = clip_above
        Normalize.__init__(self, vmin, vmax)
        
    def __call__(self, value, clip=None):
        result = super().__call__(value, clip)
        if self.clip_below is not None:
            result = np.ma.masked_where(value < self.clip_below, result)
        if self.clip_above is not None:
            result = np.ma.masked_where(value > self.clip_above, result)
        return result


def determine_format(vmin, vmax):
    # Determine appropriate format based on data range
    data_range = vmax - vmin
    if data_range < 0.01:
        return '%.4f'  # Very small range, show 4 decimal places
    elif data_range < 0.1:
        return '%.3f'  # Small range, show 3 decimal places
    elif data_range < 1:
        return '%.2f'  # Medium range, show 2 decimal places
    elif data_range < 10:
        return '%.1f'  # Larger range, show 1 decimal place
    else:
        return '%.1f'  # Large range, show 1 decimal place


def create_comparison_animation(
    true_data, 
    pred_data, 
    fig_title=None, 
    interval=100,
    fps=10,
    cmap=cmo.thermal_r,
    error_cmap=cmo.balance, 
    title='Evolution',
    save_path=None, 
    dpi=100,  # Lower DPI for faster rendering
    jpeg_quality=80,  # JPEG quality for MP4 compression
    manual_clipping=False,  # Whether to use clipping for normalization
):
    """
    Create a three-panel animation showing true, predicted, and error.
    Args:
        true_data: Ground truth data [timesteps, height, width]
        pred_data: Predicted data [timesteps, height, width]
        interval: Time between frames in milliseconds
        fps: Frames per second for saved video
        cmap: Colormap for true/pred visualization, choices: ['YlOrRd', 'viridis', cmo.thermal_r]
        error_cmap: Colormap for error visualization, choices: [cmo.balance, cmo.delta, 'RdYlBu_r', 'RdBu']
        title: Title for the animation
        save_path: Path to save the animation
        dpi: Dots per inch for figure resolution
        jpeg_quality: JPEG quality for MP4 compression
        use_clipping: Whether to use clipping for normalization
    """
    if isinstance(true_data, torch.Tensor):
        true_data = true_data.float().detach().cpu().numpy()
    if isinstance(pred_data, torch.Tensor):
        pred_data = pred_data.float().detach().cpu().numpy()
    logger.info(f"True data shape: {true_data.shape}, Predicted data shape: {pred_data.shape}")
    
    # Compute error/difference
    error_data = true_data - pred_data

    # Determine value ranges for visualization
    if manual_clipping:
        true_vmin = np.min(true_data)
        true_vmax = np.max(true_data)

        pred_vmin = np.min(pred_data)
        pred_vmax = np.max(pred_data)
    else:
        true_vmin = 0
        true_vmax = 0.7

        pred_vmin = 0
        pred_vmax = 0.7

    # Find GLOBAL min/max values across ALL frames
    # vmin = min(np.min(true_data), np.min(pred_data))
    # vmax = max(np.max(true_data), np.max(pred_data))
    # error_max = max(abs(np.min(error_data)), abs(np.max(error_data)))
    error_max = 0.7  # Set a fixed max for error visualization
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)
    plt.close()

    # Initialize plots with minimal decorations
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Use custom norm for predictions
    # pred_norm = ClippingNormalize(true_vmin, true_vmax)

    # Initialize plots
    im1 = ax1.imshow(true_data[0], cmap=cmap, vmin=true_vmin, vmax=true_vmax, animated=True)
    im2 = ax2.imshow(pred_data[0], cmap=cmap, vmin=pred_vmin, vmax=pred_vmax, animated=True)
    # im2 = ax2.imshow(pred_data[0], cmap=cmo.thermal_r, norm=pred_norm, animated=True)
    im3 = ax3.imshow(error_data[0], cmap=error_cmap, vmin=-error_max, vmax=error_max, animated=True)

    # Determine appropriate formats
    # data_format = determine_format(true_vmin, true_vmax)
    # error_format = determine_format(-error_max, error_max)

    # Add colorbars with fixed format
    cbar1 = fig.colorbar(im1, ax=ax1, format='%.2f')
    cbar2 = fig.colorbar(im2, ax=ax2, format='%.2f')
    cbar3 = fig.colorbar(im3, ax=ax3, format='%.1f')

    # Set titles
    ax1.set_title('True')
    ax2.set_title('Predicted')
    ax3.set_title('Error (True - Predicted)')

    if fig_title is not None:
        fig.suptitle(f'{title} | {fig_title} \nFrame: 1/{len(true_data)}')
    else:
        fig.suptitle(f'{title}\nFrame: 1/{len(true_data)}')
    
    def update(frame):
        # Only update the array data, not the color scaling
        im1.set_array(true_data[frame])
        im2.set_array(pred_data[frame])
        im3.set_array(error_data[frame])
        if fig_title is not None:
            fig.suptitle(f'{title} | {fig_title}\nFrame: {frame+1}/{len(true_data)}')
        else:
            fig.suptitle(f'{title}\nFrame: {frame+1}/{len(true_data)}')
        return [im1, im2, im3]
    
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(true_data),
        interval=interval,
        blit=True, 
        cache_frame_data=False  # Reduce memory usage
    )
    
    # t0 = time.time()
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
            # frames = [frame for frame in anim.frame_seq]
            # imageio.mimsave(save_path, frames, fps=fps)
            # Use imageio to save the animation as a GIF
            # with imageio.get_writer(save_path, mode='I', fps=fps, loop=0) as writer:
            #     for frame in range(len(true_data)):
            #         # Update the plot for the current frame
            #         update(frame)
                    
            #         # Save the current frame to the GIF
            #         fig.canvas.draw()
            #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            #         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #         writer.append_data(image)

        elif save_path.endswith('.mp4'):
            # For MP4, use ffmpeg with optimized settings
            writer = animation.FFMpegWriter(
                fps=fps,
                codec='h264',
                bitrate=-1,  # Let ffmpeg determine bitrate
                extra_args=[
                    '-preset', 'ultrafast',
                    '-crf', '30',  # Higher CRF = more compression
                    '-pix_fmt', 'yuv420p',  # Required for compatibility
                    '-tune', 'animation',  # Optimize for animation content
                    '-threads', 'auto'  # Use all available CPU cores
                ]
            )
            anim.save(save_path, writer=writer)
    # print(f"Saving time: {time.time() - t0:.2f}s")
    return anim


def create_density_and_ros_animation(
    true_data, 
    ros_data,  # List of arrays containing rate of spread data
    ros_timestamps=None,  # Timestamps for rate data (can be different length than true_data)
    ros_labels=None,  # Optional labels for each rate array
    ros_colors=None,  # Optional colors for each rate line
    fig_title=None, 
    interval=100,
    fps=10,
    cmap=cmo.thermal_r,
    title='Evolution and Rate of Spread',
    save_path=None, 
    dpi=100,
    jpeg_quality=80
):
    """
    Create a two-panel animation showing true density evolution and rate of spread.
    Args:
        true_data: Ground truth density data [timesteps, height, width]
        rates_data: List of arrays containing rate of spread data (can be different length than true_data)
        rates_timestamps: List of timestamp arrays for each rate array (optional)
        rates_labels: List of labels for each rate curve (optional)
        rates_colors: List of colors for each rate curve (optional)
        interval: Time between frames in milliseconds
        fps: Frames per second for saved video
        cmap: Colormap for density visualization
        title: Title for the animation
        save_path: Path to save the animation
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(true_data, torch.Tensor):
        true_data = true_data.float().detach().cpu().numpy()
    
    # Handle single rate array case by converting to list
    if not isinstance(ros_data, list):
        ros_data = [ros_data]

    # Handle timestamps
    if ros_timestamps is None:
        # Default timestamps are indices of rate data
        ros_timestamps = [np.arange(len(rate)) for rate in ros_data]
    elif not isinstance(ros_timestamps, list):
        # Convert single timestamp array to list
        ros_timestamps = [ros_timestamps]
    
    # Ensure all data is numpy arrays
    for i in range(len(ros_data)):
        if isinstance(ros_data[i], torch.Tensor):
            ros_data[i] = ros_data[i].float().detach().cpu().numpy()

    for i in range(len(ros_timestamps)):
        if isinstance(ros_timestamps[i], list):
            ros_timestamps[i] = np.array(ros_timestamps[i])
        if isinstance(ros_timestamps[i], torch.Tensor):
            ros_timestamps[i] = ros_timestamps[i].float().detach().cpu().numpy()

    # Set default labels if not provided
    if ros_labels is None:
        ros_labels = [f'Rate {i+1}' for i in range(len(ros_data))]
    
    # Set default colors if not provided
    if ros_colors is None:
        # Create a color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        ros_colors = [colors[i % len(colors)] for i in range(len(ros_data))]
    
    logger.info(f"True data shape: {true_data.shape}")
    for i, rate in enumerate(ros_data):
        logger.info(f"{ros_labels[i]} shape: {rate.shape}")
        logger.info(f"{ros_labels[i]} timestamps shape: {ros_timestamps[i].shape}")
    
    # Get min/max values for density data
    true_vmin = np.min(true_data)
    true_vmax = np.max(true_data)
    
    # Get min/max values for rates data for consistent y-axis
    ros_min = min([np.min(rate) for rate in ros_data])
    ros_max = max([np.max(rate) for rate in ros_data])
    
    # Create frame to timestamp mapping for animation
    true_data_timesteps = len(true_data)
    
    # Find the overall min and max timestamps across all rate data
    all_timestamps = np.concatenate(ros_timestamps)
    min_timestamp = np.min(all_timestamps)
    max_timestamp = np.max(all_timestamps)
    
    # Create a mapping from animation frame to timestamp
    # This assumes that true_data frames are evenly distributed across timestamp range
    frame_to_timestamp = np.linspace(min_timestamp, max_timestamp, true_data_timesteps)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi, 
                                   gridspec_kw={'width_ratios': [1, 1.2]})
    plt.close()

    # Initialize density plot
    ax1.set_xticks([])
    ax1.set_yticks([])
    im1 = ax1.imshow(true_data[0], cmap=cmap, vmin=true_vmin, vmax=true_vmax, animated=True)
    
    # Initialize rate of spread plot
    rate_lines = []
    for i, rate_data in enumerate(ros_data):
        line, = ax2.plot(ros_timestamps[i], rate_data, color=ros_colors[i], label=ros_labels[i])
        rate_lines.append(line)
    
    # Add marker for current timestep on rate plot
    time_marker, = ax2.plot([frame_to_timestamp[0]], [0], 'ro', markersize=8)
    
    # Add colorbar to density plot
    cbar1 = fig.colorbar(im1, ax=ax1, format='%.2f')
    
    # Configure rate plot
    ax2.set_xlim(min_timestamp, max_timestamp)
    y_padding = (ros_max - ros_min) * 0.05
    ax2.set_ylim(ros_min - y_padding, ros_max + y_padding)  # Add 5% padding
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rate of Spread')
    ax2.grid(True, linestyle='--', alpha=0.7)
    if len(ros_data) > 1:
        ax2.legend(loc='best')
    
    # Set titles
    ax1.set_title('Density Evolution')
    ax2.set_title('Rate of Spread')

    if fig_title is not None:
        fig.suptitle(f'{title} | {fig_title} \nFrame: 1/{true_data_timesteps}')
    else:
        fig.suptitle(f'{title}\nFrame: 1/{true_data_timesteps}')
    
    # Add tight layout for better spacing
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Add space for suptitle
    
    def update(frame):
        # Update the density data
        im1.set_array(true_data[frame])
        
        # Get the timestamp for current frame
        current_timestamp = frame_to_timestamp[frame]
        
        # Update time marker position
        time_marker.set_data([current_timestamp], [0])

        # Initialize a flag to track if we've updated the marker
        marker_updated = False
        
        # Find the y-value for the marker by interpolating from the first rate data
        # This is tricky because we need to find the y-value at the current timestamp
        try:
            # Find the closest rate data that has timestamps surrounding the current time
            for i, (timestamps, rate_data) in enumerate(zip(ros_timestamps, ros_data)):
                if timestamps[0] <= current_timestamp <= timestamps[-1]:
                    # Interpolate to find the rate value at the current timestamp
                    y_value = np.interp(current_timestamp, timestamps, rate_data)
                    time_marker.set_data([current_timestamp], [y_value])
                    marker_updated = True
                    break
            
            # If no matching timestamp range was found, place marker at the bottom
            if not marker_updated:
                time_marker.set_data([current_timestamp], [ros_min])
        except Exception as e:
            # If interpolation fails, just show the marker at timestamp with y=0
            logger.warning(f"Warning: Could not interpolate marker position: {e}")
        
        # Update title
        if fig_title is not None:
            fig.suptitle(f'{title} | {fig_title}\nFrame: {frame+1}/{true_data_timesteps}')
        else:
            fig.suptitle(f'{title}\nFrame: {frame+1}/{true_data_timesteps}')
            
        return [im1, time_marker, *rate_lines]
    
    anim = animation.FuncAnimation(
        fig, update,
        frames=true_data_timesteps,
        interval=interval,
        blit=True, 
        cache_frame_data=False  # Reduce memory usage
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            # For MP4, use ffmpeg with optimized settings
            writer = animation.FFMpegWriter(
                fps=fps,
                codec='h264',
                bitrate=-1,  # Let ffmpeg determine bitrate
                extra_args=[
                    '-preset', 'ultrafast',
                    '-crf', '30',  # Higher CRF = more compression
                    '-pix_fmt', 'yuv420p',  # Required for compatibility
                    '-tune', 'animation',  # Optimize for animation content
                    '-threads', 'auto'  # Use all available CPU cores
                ]
            )
            anim.save(save_path, writer=writer)
    
    return anim