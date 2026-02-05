"""Calculate tracking accuracy metrics."""

import numpy as np
import pandas as pd


def calculate_position_error(measured_df, ground_truth_df):
    """Merge measured and ground truth positions, compute errors.
    
    Parameters
    ----------
    measured_df : pandas.DataFrame
        Measured positions with columns: frame, x, y.
    ground_truth_df : pandas.DataFrame
        Ground truth positions with columns: frame, particle, x, y.
        
    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with additional columns:
        - x_true, y_true: Ground truth positions
        - error_x, error_y: Position errors
        - error_r: Radial error (Euclidean distance)
    """
    # For single particle tracking, we may not have particle ID in measured_df
    if 'particle' not in measured_df.columns:
        # Assume single particle (particle 0)
        measured_df = measured_df.copy()
        measured_df['particle'] = 0
    
    # Merge on frame and particle
    merged = pd.merge(
        measured_df,
        ground_truth_df,
        on=['frame', 'particle'],
        suffixes=('', '_true')
    )
    
    # Calculate errors
    merged['error_x'] = merged['x'] - merged['x_true']
    merged['error_y'] = merged['y'] - merged['y_true']
    merged['error_r'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2)
    
    return merged


def calculate_pixel_bias(error_df):
    """Bin by sub-pixel position, compute systematic errors.
    
    This function bins positions by their fractional pixel position (0 to 1)
    and computes mean errors in each bin to reveal systematic bias.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame with columns: x_true, y_true, error_x, error_y.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'x_bins': Sub-pixel x positions (bin centers)
        - 'y_bins': Sub-pixel y positions (bin centers)
        - 'bias_x': Mean x error for each x bin
        - 'bias_y': Mean y error for each y bin
        - 'count_x': Number of samples in each x bin
        - 'count_y': Number of samples in each y bin
    """
    # Extract fractional parts of true positions
    x_frac = error_df['x_true'] - np.floor(error_df['x_true'])
    y_frac = error_df['y_true'] - np.floor(error_df['y_true'])
    
    # Define bins (10 bins across [0, 1))
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin x positions and calculate mean error
    x_digitized = np.digitize(x_frac, bins) - 1
    x_digitized = np.clip(x_digitized, 0, num_bins - 1)
    
    bias_x = np.zeros(num_bins)
    count_x = np.zeros(num_bins)
    for i in range(num_bins):
        mask = x_digitized == i
        if np.sum(mask) > 0:
            bias_x[i] = np.mean(error_df.loc[mask, 'error_x'])
            count_x[i] = np.sum(mask)
    
    # Bin y positions and calculate mean error
    y_digitized = np.digitize(y_frac, bins) - 1
    y_digitized = np.clip(y_digitized, 0, num_bins - 1)
    
    bias_y = np.zeros(num_bins)
    count_y = np.zeros(num_bins)
    for i in range(num_bins):
        mask = y_digitized == i
        if np.sum(mask) > 0:
            bias_y[i] = np.mean(error_df.loc[mask, 'error_y'])
            count_y[i] = np.sum(mask)
    
    return {
        'x_bins': bin_centers,
        'y_bins': bin_centers,
        'bias_x': bias_x,
        'bias_y': bias_y,
        'count_x': count_x,
        'count_y': count_y,
    }


def calculate_summary_statistics(error_df):
    """Calculate RMSE, mean error, std dev, max error.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame with columns: error_x, error_y, error_r.
        
    Returns
    -------
    dict
        Summary statistics:
        - 'rmse_x', 'rmse_y', 'rmse_r': Root mean square errors
        - 'mean_x', 'mean_y', 'mean_r': Mean errors (bias)
        - 'std_x', 'std_y', 'std_r': Standard deviations
        - 'max_x', 'max_y', 'max_r': Maximum errors
    """
    stats = {
        'rmse_x': np.sqrt(np.mean(error_df['error_x']**2)),
        'rmse_y': np.sqrt(np.mean(error_df['error_y']**2)),
        'rmse_r': np.sqrt(np.mean(error_df['error_r']**2)),
        'mean_x': np.mean(error_df['error_x']),
        'mean_y': np.mean(error_df['error_y']),
        'mean_r': np.mean(error_df['error_r']),
        'std_x': np.std(error_df['error_x']),
        'std_y': np.std(error_df['error_y']),
        'std_r': np.std(error_df['error_r']),
        'max_x': np.max(np.abs(error_df['error_x'])),
        'max_y': np.max(np.abs(error_df['error_y'])),
        'max_r': np.max(error_df['error_r']),
    }
    return stats


def print_summary_statistics(stats, method_name=""):
    """Print formatted summary statistics to console.
    
    Parameters
    ----------
    stats : dict
        Summary statistics from calculate_summary_statistics().
    method_name : str, optional
        Name of the method for display.
    """
    if method_name:
        print(f"\n{'='*60}")
        print(f"Summary Statistics: {method_name}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
    
    print(f"RMSE (pixels):")
    print(f"  X: {stats['rmse_x']:.4f}")
    print(f"  Y: {stats['rmse_y']:.4f}")
    print(f"  Radial: {stats['rmse_r']:.4f}")
    
    print(f"\nMean Error (pixels):")
    print(f"  X: {stats['mean_x']:.4f}")
    print(f"  Y: {stats['mean_y']:.4f}")
    print(f"  Radial: {stats['mean_r']:.4f}")
    
    print(f"\nStd Dev (pixels):")
    print(f"  X: {stats['std_x']:.4f}")
    print(f"  Y: {stats['std_y']:.4f}")
    print(f"  Radial: {stats['std_r']:.4f}")
    
    print(f"\nMax Error (pixels):")
    print(f"  X: {stats['max_x']:.4f}")
    print(f"  Y: {stats['max_y']:.4f}")
    print(f"  Radial: {stats['max_r']:.4f}")
    print(f"{'='*60}\n")
