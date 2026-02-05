"""Visualization functions for tracking analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_trajectory_comparison(error_df, show=True, save_path=None):
    """Plot ground truth vs measured trajectory.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame with columns: x, y, x_true, y_true.
    show : bool, optional
        Whether to display the plot interactively. Default is True.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ground truth
    ax.plot(error_df['x_true'], error_df['y_true'], 'b-o', 
            label='Ground Truth', alpha=0.6, markersize=4)
    
    # Plot measured
    ax.plot(error_df['x'], error_df['y'], 'r-s',
            label='Measured', alpha=0.6, markersize=3)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_error_over_time(error_df, show=True, save_path=None):
    """Plot error evolution across frames.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame with columns: frame, error_x, error_y, error_r.
    show : bool, optional
        Whether to display the plot interactively. Default is True.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    # X error
    axes[0].plot(error_df['frame'], error_df['error_x'], 'b-', alpha=0.7)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('X Error (pixels)')
    axes[0].set_title('Position Error Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Y error
    axes[1].plot(error_df['frame'], error_df['error_y'], 'g-', alpha=0.7)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Y Error (pixels)')
    axes[1].grid(True, alpha=0.3)
    
    # Radial error
    axes[2].plot(error_df['frame'], error_df['error_r'], 'r-', alpha=0.7)
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Radial Error (pixels)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_pixel_bias(bias_data, show=True, save_path=None):
    """Plot TrackPy-style pixel bias plots.
    
    Creates 4 subplots showing:
    1. X error vs sub-pixel X position
    2. Y error vs sub-pixel Y position
    3. Sample count per X bin
    4. Sample count per Y bin
    
    Parameters
    ----------
    bias_data : dict
        Dictionary from calculate_pixel_bias() with keys:
        x_bins, y_bins, bias_x, bias_y, count_x, count_y.
    show : bool, optional
        Whether to display the plot interactively. Default is True.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # X bias
    axes[0, 0].plot(bias_data['x_bins'], bias_data['bias_x'], 'b-o', markersize=6)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Sub-pixel X Position')
    axes[0, 0].set_ylabel('Mean X Error (pixels)')
    axes[0, 0].set_title('Pixel Bias in X')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    
    # Y bias
    axes[0, 1].plot(bias_data['y_bins'], bias_data['bias_y'], 'g-o', markersize=6)
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Sub-pixel Y Position')
    axes[0, 1].set_ylabel('Mean Y Error (pixels)')
    axes[0, 1].set_title('Pixel Bias in Y')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    
    # X sample count
    axes[1, 0].bar(bias_data['x_bins'], bias_data['count_x'], 
                   width=0.08, color='b', alpha=0.6)
    axes[1, 0].set_xlabel('Sub-pixel X Position')
    axes[1, 0].set_ylabel('Sample Count')
    axes[1, 0].set_title('Samples per X Bin')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    
    # Y sample count
    axes[1, 1].bar(bias_data['y_bins'], bias_data['count_y'],
                   width=0.08, color='g', alpha=0.6)
    axes[1, 1].set_xlabel('Sub-pixel Y Position')
    axes[1, 1].set_ylabel('Sample Count')
    axes[1, 1].set_title('Samples per Y Bin')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_method_comparison(stats_dict, show=True, save_path=None):
    """Compare multiple methods using bar charts.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary mapping method names to statistics dictionaries.
    show : bool, optional
        Whether to display the plot interactively. Default is True.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    """
    methods = list(stats_dict.keys())
    
    rmse_r = [stats_dict[m]['rmse_r'] for m in methods]
    mean_r = [stats_dict[m]['mean_r'] for m in methods]
    std_r = [stats_dict[m]['std_r'] for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(methods))
    width = 0.6
    
    # RMSE comparison
    axes[0].bar(x, rmse_r, width, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('RMSE (pixels)')
    axes[0].set_title('Root Mean Square Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Mean error comparison
    axes[1].bar(x, mean_r, width, color='coral', alpha=0.7)
    axes[1].set_ylabel('Mean Error (pixels)')
    axes[1].set_title('Mean Radial Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Std dev comparison
    axes[2].bar(x, std_r, width, color='seagreen', alpha=0.7)
    axes[2].set_ylabel('Std Dev (pixels)')
    axes[2].set_title('Error Standard Deviation')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def show_sample_frames(frames, positions, ground_truth=None, 
                       frame_indices=None, show=True, save_path=None):
    """Display sample frames with marked positions.
    
    Parameters
    ----------
    frames : ndarray
        3D array of image frames (num_frames, height, width).
    positions : pandas.DataFrame
        Measured positions with columns: frame, x, y.
    ground_truth : pandas.DataFrame, optional
        Ground truth positions with columns: frame, x, y.
    frame_indices : list, optional
        Which frames to display. If None, shows first, middle, and last.
    show : bool, optional
        Whether to display the plot interactively. Default is True.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    """
    if frame_indices is None:
        n_frames = frames.shape[0]
        frame_indices = [0, n_frames // 2, n_frames - 1]
    
    num_samples = len(frame_indices)
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    
    if num_samples == 1:
        axes = [axes]
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        
        # Display frame
        ax.imshow(frames[frame_idx], cmap='gray', interpolation='nearest')
        
        # Mark measured positions
        measured = positions[positions['frame'] == frame_idx]
        if len(measured) > 0:
            ax.plot(measured['x'], measured['y'], 'rs', 
                   markersize=10, fillstyle='none', 
                   markeredgewidth=2, label='Measured')
        
        # Mark ground truth if provided
        if ground_truth is not None:
            truth = ground_truth[ground_truth['frame'] == frame_idx]
            if len(truth) > 0:
                ax.plot(truth['x'], truth['y'], 'b+',
                       markersize=12, markeredgewidth=2, label='Ground Truth')
        
        ax.set_title(f'Frame {frame_idx}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
