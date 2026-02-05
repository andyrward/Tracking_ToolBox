"""Cross-correlation based localization algorithm."""

import numpy as np
from scipy.signal import correlate2d
from ..core.base_locator import BaseLocator


class CrossCorrelationLocator(BaseLocator):
    """Localize particles using cross-correlation with a template.
    
    This locator uses FFT-based cross-correlation with a template image
    to find particle positions. Sub-pixel refinement is performed using
    parabolic interpolation around the correlation peak.
    
    Good for particles with consistent appearance across frames.
    
    Parameters
    ----------
    template : ndarray
        2D array representing the particle template.
    window_size : int, optional
        Search window size. If None, uses template size. Must be odd.
    upsampling_factor : int, optional
        Sub-pixel resolution factor. Default is 10.
        
    Attributes
    ----------
    template : ndarray
        The particle template.
    window_size : int
        Size of the search window.
    upsampling_factor : int
        Sub-pixel interpolation factor.
    """
    
    def __init__(self, template, window_size=None, upsampling_factor=10):
        # Determine window size
        if window_size is None:
            window_size = max(template.shape)
            if window_size % 2 == 0:
                window_size += 1
        
        super().__init__(window_size)
        
        self.template = template
        self.upsampling_factor = upsampling_factor
        
        # Normalize template
        self.template_normalized = self._normalize(template)
        
    def _normalize(self, array):
        """Normalize array to zero mean and unit variance.
        
        Parameters
        ----------
        array : ndarray
            Input array.
            
        Returns
        -------
        ndarray
            Normalized array.
        """
        mean = np.mean(array)
        std = np.std(array)
        if std < 1e-10:
            return array - mean
        return (array - mean) / std
    
    def _find_subpixel_peak(self, correlation):
        """Find sub-pixel peak using parabolic interpolation.
        
        Parameters
        ----------
        correlation : ndarray
            2D correlation map.
            
        Returns
        -------
        tuple
            (x, y) sub-pixel peak position.
        """
        # Find integer peak
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_y, peak_x = peak_idx
        
        h, w = correlation.shape
        
        # Check if peak is at edge
        if peak_x <= 0 or peak_x >= w - 1 or peak_y <= 0 or peak_y >= h - 1:
            return float(peak_x), float(peak_y)
        
        # Extract 3x3 region around peak for sub-pixel refinement
        region = correlation[peak_y-1:peak_y+2, peak_x-1:peak_x+2]
        
        if region.shape != (3, 3):
            return float(peak_x), float(peak_y)
        
        # Parabolic interpolation in x
        x_vals = region[1, :]
        if len(x_vals) == 3:
            a_x = (x_vals[2] + x_vals[0]) / 2.0 - x_vals[1]
            b_x = (x_vals[2] - x_vals[0]) / 2.0
            if abs(a_x) > 1e-10:
                x_offset = -b_x / (2.0 * a_x)
                x_offset = np.clip(x_offset, -1.0, 1.0)
            else:
                x_offset = 0.0
        else:
            x_offset = 0.0
        
        # Parabolic interpolation in y
        y_vals = region[:, 1]
        if len(y_vals) == 3:
            a_y = (y_vals[2] + y_vals[0]) / 2.0 - y_vals[1]
            b_y = (y_vals[2] - y_vals[0]) / 2.0
            if abs(a_y) > 1e-10:
                y_offset = -b_y / (2.0 * a_y)
                y_offset = np.clip(y_offset, -1.0, 1.0)
            else:
                y_offset = 0.0
        else:
            y_offset = 0.0
        
        return float(peak_x + x_offset), float(peak_y + y_offset)
    
    def _refine_single(self, window, guess_x, guess_y):
        """Refine position using cross-correlation.
        
        Parameters
        ----------
        window : ndarray
            2D array containing the search region.
        guess_x : float
            Initial x guess coordinate.
        guess_y : float
            Initial y guess coordinate.
            
        Returns
        -------
        dict
            Refinement result with keys: x, y, mass, signal, size.
        """
        # Normalize window
        window_normalized = self._normalize(window)
        
        # Compute cross-correlation
        correlation = correlate2d(
            window_normalized,
            self.template_normalized,
            mode='same',
            boundary='fill'
        )
        
        # Find sub-pixel peak
        peak_x, peak_y = self._find_subpixel_peak(correlation)
        
        # Convert from window coordinates to frame coordinates
        x_refined = guess_x - self.half_window + peak_x
        y_refined = guess_y - self.half_window + peak_y
        
        # Calculate mass and signal
        mass = np.sum(window)
        signal = np.max(window) - np.min(window)
        
        return {
            'x': float(x_refined),
            'y': float(y_refined),
            'mass': float(mass),
            'signal': float(signal),
            'size': self.window_size,
        }
