"""Basic usage example for the tracking toolbox."""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tracking_toolbox import (
    Gaussian2DLocator,
    DirectPropagator,
    IterativeTracker,
)

# Try to import TrackPy for demonstration
try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except ImportError:
    TRACKPY_AVAILABLE = False
    print("Note: TrackPy not available. Install with: pip install trackpy")


def generate_simple_test_data():
    """Generate simple test frames with a moving particle."""
    num_frames = 50
    image_shape = (100, 100)
    
    frames = []
    for i in range(num_frames):
        # Create frame with background
        frame = np.ones(image_shape) * 20
        
        # Add particle that moves diagonally
        x = 30 + i * 0.5
        y = 40 + i * 0.3
        
        # Create Gaussian spot
        y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
        spot = 200 * np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * 2.0**2))
        
        frame += spot
        
        # Add noise
        frame = np.random.poisson(frame).astype(float)
        frame += np.random.normal(0, np.sqrt(20), image_shape)
        
        frames.append(frame)
    
    return np.array(frames)


def main():
    """Demonstrate basic usage of the tracking toolbox."""
    print("="*70)
    print("Tracking Toolbox - Basic Usage Example")
    print("="*70)
    
    # Generate test data
    print("\n1. Generating test data...")
    frames = generate_simple_test_data()
    print(f"   Generated {len(frames)} frames of size {frames.shape[1:]}")
    
    # Set up tracker
    print("\n2. Setting up tracker...")
    locator = Gaussian2DLocator(window_size=11, sigma_guess=2.0)
    propagator = DirectPropagator()
    tracker = IterativeTracker(locator, propagator)
    print("   Using Gaussian2DLocator + DirectPropagator")
    
    # Track particles
    print("\n3. Tracking particles...")
    initial_coords = [(30, 40)]  # Starting position
    features = tracker.track(frames, initial_coords)
    print(f"   Tracked {len(features)} positions")
    print("\n   Sample results:")
    print(features.head(10))
    
    # Demonstrate TrackPy integration
    if TRACKPY_AVAILABLE:
        print("\n4. TrackPy Integration:")
        print("   The output DataFrame is TrackPy-compatible!")
        print("   You can now use TrackPy functions:")
        print()
        print("   # Link trajectories (not needed for single particle)")
        print("   trajectories = tp.link(features, search_range=10, memory=3)")
        print()
        print("   # Filter short trajectories")
        print("   trajectories = tp.filter_stubs(trajectories, threshold=10)")
        print()
        print("   # Compute and subtract drift")
        print("   drift = tp.compute_drift(trajectories)")
        print("   trajectories = tp.subtract_drift(trajectories, drift)")
    else:
        print("\n4. TrackPy Integration:")
        print("   Install TrackPy to use advanced features:")
        print("   pip install trackpy")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    
    print("\nNext steps:")
    print("  - Try different locators: Parabola2DLocator, CrossCorrelationLocator")
    print("  - Try MaxIntensityPropagator for faster-moving particles")
    print("  - Use TrackPy for trajectory linking and drift correction")
    print("  - Run the full test suite: python tests/test_tracking_accuracy.py")
    print()


if __name__ == "__main__":
    np.random.seed(42)
    main()
