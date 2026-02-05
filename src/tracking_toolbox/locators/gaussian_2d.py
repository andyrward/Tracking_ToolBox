"""Gaussian 2D fitting localization algorithm."""

import time
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
    debug_mode : bool, optional
        Enable debug mode to collect performance statistics. Default is False.
        
    Attributes
    ----------
    window_size : int
        Size of the extraction window.
    sigma_guess : float
        Initial guess for sigma parameter.
    min_mass : float
        Minimum mass threshold.
    debug_mode : bool
        Whether debug mode is enabled.
    _debug_times : list
        Debug: timing data for each localization (if debug_mode=True).
    _debug_failures : list
        Debug: failure information (if debug_mode=True).
    _debug_strategy_used : list
        Debug: which strategy was used for each localization (if debug_mode=True).
    """
    
    def __init__(self, window_size=11, sigma_guess=2.0, min_mass=100, debug_mode=False):
        super().__init__(window_size)
        self.sigma_guess = sigma_guess
        self.min_mass = min_mass
        self.debug_mode = debug_mode
        
        if self.debug_mode:
            self._debug_times = []
            self._debug_failures = []
            self._debug_strategy_used = []
        
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
    
    def _estimate_parameters(self, window):
        """Improved parameter estimation using statistical moments and robust estimators.
        
        Parameters
        ----------
        window : ndarray
            2D array containing the particle image.
        
        Returns
        -------
        tuple
            (amplitude_guess, x0_guess, y0_guess, sigma_guess, bg_guess)
        """
        # Background from edge pixels (more robust to noise)
        edge_pixels = np.concatenate([
            window[0, :], window[-1, :],
            window[:, 0], window[:, -1]
        ])
        bg_guess = np.median(edge_pixels)
        
        # Center region for finding peak (reduce noise influence)
        center = self.half_window
        margin = 2
        center_region = window[center-margin:center+margin+1, 
                              center-margin:center+margin+1]
        amplitude_guess = np.max(center_region) - bg_guess
        
        # Intensity-weighted centroid for better initial position
        threshold = bg_guess + 0.1 * amplitude_guess
        mask = window > threshold
        if np.any(mask):
            y_idx, x_idx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
            weights = (window - bg_guess) * mask
            total_weight = np.sum(weights)
            if total_weight > 0:
                x0_guess = np.sum(x_idx * weights) / total_weight
                y0_guess = np.sum(y_idx * weights) / total_weight
            else:
                x0_guess = self.half_window
                y0_guess = self.half_window
        else:
            x0_guess = self.half_window
            y0_guess = self.half_window
        
        # Sigma from second moment of intensity distribution
        y_idx, x_idx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        weights = (window - bg_guess).clip(0)
        total_weight = np.sum(weights)
        if total_weight > 0:
            sigma_x = np.sqrt(np.sum(weights * (x_idx - x0_guess)**2) / total_weight)
            sigma_y = np.sqrt(np.sum(weights * (y_idx - y0_guess)**2) / total_weight)
            sigma_guess = (sigma_x + sigma_y) / 2.0
            # Clamp to reasonable range
            sigma_guess = np.clip(sigma_guess, 0.5, self.window_size / 2)
        else:
            sigma_guess = self.sigma_guess
        
        return amplitude_guess, x0_guess, y0_guess, sigma_guess, bg_guess
    
    def _refine_single_optimized(self, window, guess_x, guess_y):
        """Optimized two-stage Gaussian fitting.
        
        Stage 1: Estimate parameters from moments
        Stage 2: Fit Gaussian to high-SNR central region
        
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
        # Stage 1: Get good initial estimates
        amp, x0, y0, sigma, bg = self._estimate_parameters(window)
        
        # Validate estimates
        if amp < self.min_mass / (2 * np.pi * sigma**2 + 1e-10):
            return {'failed': True}
        
        # Stage 2: Fit only central high-SNR region
        fit_size = min(7, self.window_size)  # 7x7 region
        half_fit = fit_size // 2
        cx, cy = int(np.round(x0)), int(np.round(y0))
        
        # Ensure fit region is within bounds
        cx = np.clip(cx, half_fit, window.shape[1] - half_fit - 1)
        cy = np.clip(cy, half_fit, window.shape[0] - half_fit - 1)
        
        # Extract fit region
        y_start = cy - half_fit
        y_end = cy + half_fit + 1
        x_start = cx - half_fit
        x_end = cx + half_fit + 1
        
        fit_window = window[y_start:y_end, x_start:x_end]
        y_coords, x_coords = np.mgrid[0:fit_window.shape[0], 0:fit_window.shape[1]]
        
        # Adjust initial guesses for fit window coordinates
        x0_fit = x0 - x_start
        y0_fit = y0 - y_start
        
        p0 = [amp, x0_fit, y0_fit, sigma, bg]
        
        # Adaptive bounds based on estimates
        bounds = (
            [0.3*amp, x0_fit-1.5, y0_fit-1.5, 0.5, max(0, 0.5*bg)],
            [3*amp, x0_fit+1.5, y0_fit+1.5, self.window_size/2, 3*bg]
        )
        
        try:
            # Fit with adaptive tolerances
            popt, pcov = curve_fit(
                self._gaussian_2d,
                (x_coords, y_coords),
                fit_window.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=500,
                ftol=1e-4,
                xtol=1e-4
            )
            
            amplitude, x0_fit, y0_fit, sigma, bg = popt
            
            # Convert back to full window coordinates
            x0 = x0_fit + x_start
            y0 = y0_fit + y_start
            
            # Calculate mass (integrated intensity)
            mass = amplitude * 2 * np.pi * sigma**2
            
            # Enhanced validation
            if (mass < self.min_mass or 
                sigma < 0.5 or 
                sigma > self.window_size or
                amplitude < 0 or
                x0 < 0 or x0 >= window.shape[1] or
                y0 < 0 or y0 >= window.shape[0]):
                return {'failed': True}
            
            # Convert from window coordinates to frame coordinates
            # Window is centered at integer pixel position
            ix = int(round(guess_x))
            iy = int(round(guess_y))
            x_refined = ix + (x0 - self.half_window)
            y_refined = iy + (y0 - self.half_window)
            
            if self.debug_mode:
                self._debug_strategy_used.append('optimized')
            
            return {
                'x': float(x_refined),
                'y': float(y_refined),
                'mass': float(mass),
                'signal': float(amplitude),
                'size': float(sigma),
            }
            
        except (RuntimeError, ValueError):
            return {'failed': True}
    
    def _refine_single_original(self, window, guess_x, guess_y):
        """Original implementation as fallback.
        
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
                return {'failed': True}
            
            # Convert from window coordinates to frame coordinates
            # Window is centered at integer pixel position
            ix = int(round(guess_x))
            iy = int(round(guess_y))
            x_refined = ix + (x0 - self.half_window)
            y_refined = iy + (y0 - self.half_window)
            
            if self.debug_mode:
                self._debug_strategy_used.append('original')
            
            return {
                'x': float(x_refined),
                'y': float(y_refined),
                'mass': float(mass),
                'signal': float(amplitude),
                'size': float(sigma),
            }
            
        except (RuntimeError, ValueError):
            # Fit failed
            return {'failed': True}
    
    def _refine_with_retry(self, window, guess_x, guess_y):
        """Attempt fitting with multiple strategies before falling back to centroid.
        
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
        # Strategy 1: Optimized 7x7 fit
        result = self._refine_single_optimized(window, guess_x, guess_y)
        if not result.get('failed', False) and 'x' in result:
            return result
        
        # Strategy 2: Full window fit with original parameters
        try:
            result = self._refine_single_original(window, guess_x, guess_y)
            if not result.get('failed', False) and 'x' in result:
                return result
        except (RuntimeError, ValueError, Exception) as e:
            # Fit failed, continue to fallback
            pass
        
        # Strategy 3: Centroid fallback
        if self.debug_mode:
            self._debug_strategy_used.append('centroid')
        return self._centroid_fallback(window, guess_x, guess_y)
    
    def get_debug_stats(self):
        """Return performance statistics if debug mode is enabled.
        
        Returns
        -------
        dict or None
            Dictionary with timing and failure statistics, or None if debug mode is disabled.
        """
        if not self.debug_mode:
            return None
        
        try:
            import pandas as pd
            strategy_dist = pd.Series(self._debug_strategy_used).value_counts().to_dict() if self._debug_strategy_used else {}
        except ImportError:
            # Pandas not available, use basic dict counting
            strategy_dist = {}
            for strategy in self._debug_strategy_used:
                strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
        
        return {
            'mean_time': np.mean(self._debug_times) if self._debug_times else 0,
            'median_time': np.median(self._debug_times) if self._debug_times else 0,
            'failure_rate': len(self._debug_failures) / len(self._debug_times) if self._debug_times else 0,
            'strategy_distribution': strategy_dist
        }
    
    def _refine_single(self, window, guess_x, guess_y):
        """Refine position by fitting 2D Gaussian (optimized version).
        
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
        if self.debug_mode:
            t0 = time.time()
        
        # Use optimized implementation with retry strategy
        result = self._refine_with_retry(window, guess_x, guess_y)
        
        if self.debug_mode:
            dt = time.time() - t0
            self._debug_times.append(dt)
            
            if result.get('failed', False):
                self._debug_failures.append({
                    'window_stats': {
                        'max': np.max(window),
                        'min': np.min(window),
                        'std': np.std(window),
                        'snr': (np.max(window) - np.min(window)) / (np.std(window) + 1e-10)
                    }
                })
        
        return result
