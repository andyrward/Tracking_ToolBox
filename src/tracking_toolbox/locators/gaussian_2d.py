"""Gaussian 2D fitting localization algorithm."""

import numpy as np
from scipy.optimize import curve_fit
from ..core.base_locator import BaseLocator


class Gaussian2DLocator(BaseLocator):
    """Localize particles by fitting 2D Gaussian profiles.
    
    This locator fits a symmetric 2D Gaussian to the particle intensity:
        I(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)) + bg
    
    Parameters
    ----------
    window_size : int, optional
        Size of extraction window. Must be odd. Default is 11.
    sigma_guess : float, optional
        Initial guess for Gaussian width. Default is 2.0.
    min_mass : float, optional
        Minimum integrated intensity. Default is 100.
        
    Attributes
    ----------
    window_size : int
        Size of the extraction window.
    sigma_guess : float
        Initial guess for sigma parameter.
    min_mass : float
        Minimum mass threshold.
    """
    
    def __init__(self, window_size=11, sigma_guess=2.0, min_mass=100):
        super().__init__(window_size)
        self.sigma_guess = sigma_guess
        self.min_mass = min_mass
        
    def _gaussian_2d(self, coords, amplitude, x0, y0, sigma, bg):
        """2D Gaussian function for fitting.
        
        Parameters
        ----------
        coords : tuple
            (x, y) coordinate grids.
        amplitude : float
            Peak amplitude above background.
        x0, y0 : float
            Center coordinates.
        sigma : float
            Width parameter.
        bg : float
            Background level.
            
        Returns
        -------
        ndarray
            Flattened Gaussian values.
        """
        x, y = coords
        gauss = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + bg
        return gauss.ravel()
    
    def _refine_single(self, window, guess_x, guess_y):
        """Refine position by fitting 2D Gaussian.
        
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
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        
        # Initial parameter guesses
        amplitude_guess = np.max(window) - np.min(window)
        bg_guess = np.min(window)
        x0_guess = self.half_window
        y0_guess = self.half_window
        
        # Initial parameters: [amplitude, x0, y0, sigma, bg]
        p0 = [amplitude_guess, x0_guess, y0_guess, self.sigma_guess, bg_guess]
        
        # Bounds for parameters
        bounds = (
            [0, 0, 0, 0.5, 0],  # Lower bounds
            [np.inf, window.shape[1], window.shape[0], window.shape[0], np.inf]  # Upper bounds
        )
        
        try:
            # Fit Gaussian
            popt, _ = curve_fit(
                self._gaussian_2d,
                (x_coords, y_coords),
                window.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=1000
            )
            
            amplitude, x0, y0, sigma, bg = popt
            
            # Calculate mass (integrated intensity)
            mass = amplitude * 2 * np.pi * sigma**2
            
            # Check if fit is reasonable
            if mass < self.min_mass or sigma < 0.5 or sigma > self.window_size:
                # Fallback to centroid
                return self._centroid_fallback(window, guess_x, guess_y)
            
            # Convert from window coordinates to frame coordinates
            x_refined = guess_x - self.half_window + x0
            y_refined = guess_y - self.half_window + y0
            
            return {
                'x': float(x_refined),
                'y': float(y_refined),
                'mass': float(mass),
                'signal': float(amplitude),
                'size': float(sigma),
            }
            
        except (RuntimeError, ValueError):
            # Fit failed, use centroid fallback
            return self._centroid_fallback(window, guess_x, guess_y)
