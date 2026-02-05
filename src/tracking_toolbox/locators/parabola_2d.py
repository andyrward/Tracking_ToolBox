"""Parabola 2D fitting localization algorithm."""

import numpy as np
from ..core.base_locator import BaseLocator

# Constants
MIN_PARABOLA_CURVATURE = 1e-10  # Minimum curvature to consider parabola valid


class Parabola2DLocator(BaseLocator):
    """Localize particles by fitting 2D parabola to peak.
    
    This locator finds the maximum intensity pixel and fits 1D parabolas
    in the x and y directions through a 3x3 region around the maximum.
    This is faster than Gaussian fitting and works well for well-defined peaks.
    
    The parabola is fit through 3 points to find the peak location:
        x_peak = -b/(2a) where the parabola is y = ax^2 + bx + c
    
    Parameters
    ----------
    window_size : int, optional
        Size of extraction window. Must be odd. Default is 11.
    min_mass : float, optional
        Minimum integrated intensity. Default is 100.
        
    Attributes
    ----------
    window_size : int
        Size of the extraction window.
    min_mass : float
        Minimum mass threshold.
    """
    
    def __init__(self, window_size=11, min_mass=100):
        super().__init__(window_size)
        self.min_mass = min_mass
        
    def _fit_parabola_1d(self, values):
        """Fit parabola through 3 points and find peak.
        
        Given three points at x = [-1, 0, 1] with values [v0, v1, v2],
        fit parabola y = ax^2 + bx + c and find peak at x = -b/(2a).
        
        Parameters
        ----------
        values : array-like
            Three intensity values.
            
        Returns
        -------
        float
            Peak offset from center point. Returns 0 if fit fails.
        """
        if len(values) != 3:
            return 0.0
            
        v0, v1, v2 = values
        
        # Coefficients for parabola through (-1, v0), (0, v1), (1, v2)
        # y = ax^2 + bx + c
        # c = v1
        # a + b + c = v2  =>  a + b = v2 - v1
        # a - b + c = v0  =>  a - b = v0 - v1
        # Solving: a = (v2 + v0)/2 - v1, b = (v2 - v0)/2
        
        a = (v2 + v0) / 2.0 - v1
        b = (v2 - v0) / 2.0
        
        # Peak at x = -b/(2a)
        if abs(a) < MIN_PARABOLA_CURVATURE:
            # Nearly flat, no refinement
            return 0.0
            
        peak_offset = -b / (2.0 * a)
        
        # Sanity check: peak should be within [-1, 1]
        if abs(peak_offset) > 1.5:
            return 0.0
            
        return peak_offset
    
    def _refine_single(self, window, guess_x, guess_y):
        """Refine position by fitting parabolas.
        
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
            Refinement result with keys: x, y, mass, signal, size.
        """
        # Find maximum pixel in window
        max_idx = np.unravel_index(np.argmax(window), window.shape)
        max_y, max_x = max_idx
        
        # Check if maximum is at window edge
        if (max_x <= 0 or max_x >= window.shape[1] - 1 or
            max_y <= 0 or max_y >= window.shape[0] - 1):
            # Can't fit parabola at edge, use centroid fallback
            return self._centroid_fallback(window, guess_x, guess_y)
        
        # Extract 3x3 region around maximum
        region = window[max_y-1:max_y+2, max_x-1:max_x+2]
        
        if region.shape != (3, 3):
            # Edge case, use centroid
            return self._centroid_fallback(window, guess_x, guess_y)
        
        # Fit parabola in x direction (middle row)
        x_values = region[1, :]
        x_offset = self._fit_parabola_1d(x_values)
        
        # Fit parabola in y direction (middle column)
        y_values = region[:, 1]
        y_offset = self._fit_parabola_1d(y_values)
        
        # Refined position in window coordinates
        x_refined_window = max_x + x_offset
        y_refined_window = max_y + y_offset
        
        # Convert to frame coordinates
        x_refined = guess_x - self.half_window + x_refined_window
        y_refined = guess_y - self.half_window + y_refined_window
        
        # Calculate mass and signal
        mass = np.sum(window)
        signal = np.max(window) - np.min(window)
        
        # Check minimum mass threshold
        if mass < self.min_mass:
            return self._centroid_fallback(window, guess_x, guess_y)
        
        return {
            'x': float(x_refined),
            'y': float(y_refined),
            'mass': float(mass),
            'signal': float(signal),
            'size': self.window_size,
        }
