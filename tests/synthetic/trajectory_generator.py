"""Generate synthetic particle trajectories."""

import numpy as np
import pandas as pd
from .spot_generator import generate_spot_with_noise


def generate_diagonal_trajectory(start_pos, velocity, num_frames):
    """Generate linear trajectory with constant velocity.
    
    Parameters
    ----------
    start_pos : tuple
        (x, y) starting position.
    velocity : tuple
        (vx, vy) velocity in pixels per frame.
    num_frames : int
        Number of frames.
        
    Returns
    -------
    ndarray
        Nx2 array of (x, y) positions for each frame.
    """
    x0, y0 = start_pos
    vx, vy = velocity
    
    frames = np.arange(num_frames)
    x = x0 + vx * frames
    y = y0 + vy * frames
    
    return np.column_stack([x, y])


def generate_multiple_trajectories(start_positions, velocities, num_frames):
    """Generate multiple particle trajectories.
    
    Parameters
    ----------
    start_positions : list of tuple
        List of (x, y) starting positions.
    velocities : list of tuple
        List of (vx, vy) velocities.
    num_frames : int
        Number of frames.
        
    Returns
    -------
    list of ndarray
        List of trajectories, each Nx2 array.
    """
    trajectories = []
    
    for start_pos, velocity in zip(start_positions, velocities):
        traj = generate_diagonal_trajectory(start_pos, velocity, num_frames)
        trajectories.append(traj)
    
    return trajectories


def generate_frame_sequence(trajectories, image_shape, amplitude, sigma,
                           background, gaussian_noise_sigma=None):
    """Create image sequence from trajectories.
    
    Parameters
    ----------
    trajectories : list of ndarray
        List of particle trajectories. Each trajectory is Nx2 array of (x, y)
        positions for N frames.
    image_shape : tuple
        (height, width) of output images.
    amplitude : float or list
        Peak amplitude(s) for particles.
    sigma : float or list
        Gaussian width parameter(s).
    background : float
        Background intensity level.
    gaussian_noise_sigma : float, optional
        Standard deviation for Gaussian noise. Default is sqrt(background).
        
    Returns
    -------
    ndarray
        3D array (num_frames, height, width) of image frames.
    pandas.DataFrame
        Ground truth positions with columns: frame, particle, x, y.
    """
    if not trajectories:
        raise ValueError("No trajectories provided")
    
    num_frames = len(trajectories[0])
    num_particles = len(trajectories)
    
    # Handle scalar amplitude and sigma
    if not isinstance(amplitude, (list, tuple, np.ndarray)):
        amplitude = [amplitude] * num_particles
    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = [sigma] * num_particles
    
    # Default Gaussian noise
    if gaussian_noise_sigma is None:
        gaussian_noise_sigma = np.sqrt(background)
    
    # Initialize frame array
    frames = np.zeros((num_frames, image_shape[0], image_shape[1]))
    
    # Ground truth data
    ground_truth = []
    
    # Generate each frame
    for frame_idx in range(num_frames):
        frame = np.ones(image_shape) * background
        
        # Add each particle to this frame
        for particle_idx, trajectory in enumerate(trajectories):
            x, y = trajectory[frame_idx]
            
            # Generate particle spot (no background, as frame already has it)
            spot = generate_spot_with_noise(
                image_shape,
                x, y,
                amplitude[particle_idx],
                sigma[particle_idx],
                0,  # No background in spot (frame already has background)
                poisson_noise=False,  # Add noise later to combined frame
                gaussian_noise_sigma=None
            )
            
            # Add spot to frame (spot already has zero background)
            frame += spot
            
            # Record ground truth
            ground_truth.append({
                'frame': frame_idx,
                'particle': particle_idx,
                'x': x,
                'y': y
            })
        
        # Add noise to complete frame
        frame = np.random.poisson(frame).astype(float)
        frame = frame + np.random.normal(0, gaussian_noise_sigma, image_shape)
        
        frames[frame_idx] = frame
    
    ground_truth_df = pd.DataFrame(ground_truth)
    
    return frames, ground_truth_df
