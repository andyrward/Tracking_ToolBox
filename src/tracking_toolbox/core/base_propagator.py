"""Base class for coordinate propagation between frames."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BasePropagator(ABC):
    """Abstract base class for generating guess coordinates.
    
    Propagators take results from frame N and generate guess coordinates
    for frame N+1. Different strategies can be implemented for different
    tracking scenarios.
    """
    
    @abstractmethod
    def propagate(self, current_frame, previous_positions, frame_number):
        """Generate guess coordinates for the next frame.
        
        Parameters
        ----------
        current_frame : ndarray
            2D image of the current frame.
        previous_positions : pandas.DataFrame or array-like
            Positions from the previous frame. If DataFrame, should have
            'x' and 'y' columns. If array-like, should be Nx2 array.
        frame_number : int
            Current frame number (for reference).
            
        Returns
        -------
        array-like
            Guess coordinates as list of (x, y) tuples or Nx2 array.
        """
        pass
    
    def _positions_to_array(self, positions):
        """Convert positions to numpy array.
        
        Parameters
        ----------
        positions : pandas.DataFrame or array-like
            Position data.
            
        Returns
        -------
        ndarray
            Nx2 array of (x, y) coordinates.
        """
        if isinstance(positions, pd.DataFrame):
            return positions[['x', 'y']].values
        else:
            return np.array(positions)
