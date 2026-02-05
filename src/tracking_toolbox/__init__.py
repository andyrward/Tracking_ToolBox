"""Tracking Toolbox - Custom particle tracking with TrackPy integration."""

__version__ = "0.1.0"

# Import main classes for convenient access
from .locators.gaussian_2d import Gaussian2DLocator
from .locators.parabola_2d import Parabola2DLocator
from .locators.cross_correlation import CrossCorrelationLocator

from .propagators.direct import DirectPropagator
from .propagators.max_intensity import MaxIntensityPropagator

from .trackers.iterative_tracker import IterativeTracker

__all__ = [
    'Gaussian2DLocator',
    'Parabola2DLocator',
    'CrossCorrelationLocator',
    'DirectPropagator',
    'MaxIntensityPropagator',
    'IterativeTracker',
]