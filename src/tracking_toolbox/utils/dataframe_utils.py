"""Utility functions for DataFrame operations."""

import pandas as pd
import numpy as np


def add_particle_ids(df, num_particles):
    """Add particle ID column to tracking results.
    
    Assumes particles are tracked in order and all frames have the same
    number of particles.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Tracking results with 'frame' column.
    num_particles : int
        Number of particles tracked.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'particle' column.
    """
    if len(df) == 0:
        df['particle'] = []
        return df
    
    frames = df['frame'].unique()
    num_frames = len(frames)
    
    # Create particle IDs that repeat for each frame
    particle_ids = np.tile(np.arange(num_particles), num_frames)
    
    # Trim to match dataframe length (in case some frames have fewer particles)
    particle_ids = particle_ids[:len(df)]
    
    df['particle'] = particle_ids
    return df


def ensure_trackpy_columns(df):
    """Ensure DataFrame has all required TrackPy columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with at minimum: frame, x, y columns.
    """
    required_cols = ['frame', 'x', 'y']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")
    
    return df
