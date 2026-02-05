"""Generate synthetic spots for testing."""

import numpy as np


def generate_gaussian_spot(image_shape, x, y, amplitude, sigma, background=0):
    """Create 2D Gaussian spot at sub-pixel position.
    
    Parameters
    ----------
    image_shape : tuple
        (height, width) of output image.
    x : float
        X coordinate (column) of spot center.
    y : float
        Y coordinate (row) of spot center.
    amplitude : float
        Peak amplitude above background.
    sigma : float
        Gaussian width parameter.
    background : float, optional
        Background intensity level. Default is 0.
        
    Returns
    -------
    ndarray
        2D image with Gaussian spot.
    """
    h, w = image_shape
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Calculate Gaussian
    gaussian = amplitude * np.exp(
        -((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2)
    ) + background
    
    return gaussian


def add_poisson_noise(image):
    """Add Poisson (shot) noise to image.
    
    Parameters
    ----------
    image : ndarray
        Input image (must be non-negative).
        
    Returns
    -------
    ndarray
        Image with Poisson noise added.
    """
    # Poisson noise has variance equal to the mean
    return np.random.poisson(image).astype(float)


def add_gaussian_noise(image, sigma):
    """Add Gaussian (readout) noise to image.
    
    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation of Gaussian noise.
        
    Returns
    -------
    ndarray
        Image with Gaussian noise added.
    """
    noise = np.random.normal(0, sigma, image.shape)
    return image + noise


def generate_spot_with_noise(image_shape, x, y, amplitude, sigma, background,
                            poisson_noise=True, gaussian_noise_sigma=None):
    """Generate Gaussian spot with realistic noise.
    
    Parameters
    ----------
    image_shape : tuple
        (height, width) of output image.
    x : float
        X coordinate of spot center.
    y : float
        Y coordinate of spot center.
    amplitude : float
        Peak amplitude above background.
    sigma : float
        Gaussian width parameter.
    background : float
        Background intensity level.
    poisson_noise : bool, optional
        Whether to add Poisson noise. Default is True.
    gaussian_noise_sigma : float, optional
        Standard deviation for Gaussian noise. If None, no Gaussian noise added.
        Common choice: sqrt(background) for readout noise.
        
    Returns
    -------
    ndarray
        2D image with spot and noise.
    """
    # Generate clean spot
    image = generate_gaussian_spot(image_shape, x, y, amplitude, sigma, background)
    
    # Add Poisson noise (shot noise)
    if poisson_noise:
        image = add_poisson_noise(image)
    
    # Add Gaussian noise (readout noise)
    if gaussian_noise_sigma is not None:
        image = add_gaussian_noise(image, gaussian_noise_sigma)
    
    return image
