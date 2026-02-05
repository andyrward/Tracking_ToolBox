"""Base class for particle localization algorithms."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseLocator(ABC):
    """Abstract base class for sub-pixel particle localization algorithms.
    
    Locators refine guess coordinates to sub-pixel accuracy by analyzing
    small windows around each particle. All locators return TrackPy-compatible
    DataFrames with columns: x, y, mass, signal, size.
    
    Parameters
    ----------
    window_size : int, optional
        Size of extraction window in pixels. Must be odd. Default is 11.
        
    Attributes
    ----------
    window_size : int
        Size of the extraction window.
    half_window : int
        Half of window_size for convenient indexing.
    """
    
    def __init__(self, window_size=11):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.window_size = window_size
        self.half_window = window_size // 2
        
    @abstractmethod
    def _refine_single(self, window, guess_x, guess_y):
        """Refine a single particle position.
        
        This method should be implemented by subclasses to perform the actual
        localization algorithm on a single particle window.
        
        Parameters
        ----------
        window : ndarray
            2D array containing the particle image.
        guess_x : float
            Initial x guess coordinate (center of window).
        guess_y : float
            Initial y guess coordinate (center of window).
            
        Returns
        -------
        dict
            Dictionary with at minimum: 'x', 'y', 'mass', 'signal', 'size'.
            If localization fails, should return None or dict with failed=True.
        """
        pass
    
    def locate(self, frame, coordinates, return_failed=False):
        """Refine multiple particles in a single frame.
        
        Parameters
        ----------
        frame : ndarray
            2D image frame.
        coordinates : array-like
            List of (x, y) tuples or 2-column array of guess coordinates.
        return_failed : bool, optional
            If True, include failed localizations in output. Default is False.
            
        Returns
        -------
        pandas.DataFrame
            TrackPy-compatible DataFrame with columns: x, y, mass, signal, size.
        """
        results = []
        
        for coord in coordinates:
            if isinstance(coord, (tuple, list)):
                x, y = coord
            else:
                x, y = coord[0], coord[1]
                
            # Extract window around guess position
            window = self._extract_window(frame, x, y)
            
            if window is None:
                # Position is outside valid region
                if return_failed:
                    results.append({
                        'x': x, 'y': y, 'mass': 0, 'signal': 0, 
                        'size': 0, 'failed': True
                    })
                continue
                
            # Refine position
            result = self._refine_single(window, x, y)
            
            if result is None or result.get('failed', False):
                if return_failed:
                    results.append({
                        'x': x, 'y': y, 'mass': 0, 'signal': 0,
                        'size': 0, 'failed': True
                    })
                continue
                
            results.append(result)
            
        if not results:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['x', 'y', 'mass', 'signal', 'size'])
            
        return pd.DataFrame(results)
    
    def _extract_window(self, frame, x, y):
        """Extract window around position with boundary handling.
        
        Parameters
        ----------
        frame : ndarray
            2D image frame.
        x : float
            X coordinate (column).
        y : float
            Y coordinate (row).
            
        Returns
        -------
        ndarray or None
            Extracted window, or None if position is too close to boundary.
        """
        h, w = frame.shape
        
        # Convert to integer pixel coordinates
        ix = int(round(x))
        iy = int(round(y))
        
        # Check if window fits within frame
        if (ix - self.half_window < 0 or ix + self.half_window >= w or
            iy - self.half_window < 0 or iy + self.half_window >= h):
            return None
            
        # Extract window
        window = frame[iy - self.half_window:iy + self.half_window + 1,
                      ix - self.half_window:ix + self.half_window + 1]
        
        return window
    
    def _centroid_fallback(self, window, guess_x, guess_y):
        """Calculate centroid as fallback for failed fits.
        
        Parameters
        ----------
        window : ndarray
            2D array containing the particle image.
        guess_x : float
            Initial x guess coordinate.
        guess_y : float
            Initial y guess coordinate.
            
        Returns
        -------
        dict
            Refinement result with centroid position.
        """
        # Subtract minimum to avoid negative weights
        window_shifted = window - np.min(window)
        
        total = np.sum(window_shifted)
        if total <= 0:
            return {
                'x': guess_x, 'y': guess_y, 'mass': 0,
                'signal': 0, 'size': 0, 'failed': True
            }
        
        y_coords, x_coords = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        
        cx = np.sum(x_coords * window_shifted) / total
        cy = np.sum(y_coords * window_shifted) / total
        
        # Convert from window coordinates to frame coordinates
        x_refined = guess_x - self.half_window + cx
        y_refined = guess_y - self.half_window + cy
        
        mass = np.sum(window)
        signal = np.max(window) - np.min(window)
        
        return {
            'x': x_refined,
            'y': y_refined,
            'mass': float(mass),
            'signal': float(signal),
            'size': self.window_size,
        }
