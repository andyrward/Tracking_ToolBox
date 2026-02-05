# Tracking Toolbox

A Python particle tracking toolbox that provides custom localization algorithms while integrating seamlessly with [TrackPy](http://soft-matter.github.io/trackpy/) for trajectory linking, pixel bias correction, and drift correction.

## Overview

This toolbox focuses on **sub-pixel particle localization** with multiple algorithms, while leveraging TrackPy's robust features for:
- Trajectory linking across frames
- Handling particle disappearances and reappearances
- Drift correction
- Filtering spurious trajectories

### Philosophy

- **Custom Localization**: Implement specialized localization algorithms (Gaussian fitting, parabolic interpolation, cross-correlation)
- **TrackPy Integration**: All locators output TrackPy-compatible DataFrames
- **Modular Design**: Mix and match locators and propagators for different scenarios
- **Sub-pixel Accuracy**: Designed for high-precision measurements (< 0.1 pixel RMSE)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/andyrward/Tracking_ToolBox.git
cd Tracking_ToolBox

# Install in development mode
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Pandas >= 1.3
- Matplotlib >= 3.4
- TrackPy >= 0.5

All dependencies are automatically installed with the package.

## Quick Start

```python
from tracking_toolbox import (
    Gaussian2DLocator,
    DirectPropagator,
    IterativeTracker
)
import trackpy as tp

# Set up custom tracker
locator = Gaussian2DLocator(window_size=11, sigma_guess=2.0)
propagator = DirectPropagator()
tracker = IterativeTracker(locator, propagator)

# Track particles
initial_coords = [(100, 150), (200, 250)]
features = tracker.track(frames, initial_coords)

# Use TrackPy for linking and analysis
trajectories = tp.link(features, search_range=10, memory=3)
trajectories = tp.filter_stubs(trajectories, threshold=5)
drift = tp.compute_drift(trajectories)
trajectories = tp.subtract_drift(trajectories, drift)
```

## Architecture

### Core Components

#### Base Classes (`src/tracking_toolbox/core/`)

- **BaseLocator**: Abstract base class for localization algorithms
  - Takes guess coordinates and refines to sub-pixel accuracy
  - Operates on small windows around each particle
  - Returns TrackPy-compatible DataFrames
  
- **BasePropagator**: Abstract base class for coordinate propagation
  - Generates guess coordinates for frame N+1 based on frame N
  - Supports multiple propagation strategies

#### Localization Algorithms (`src/tracking_toolbox/locators/`)

1. **Gaussian2DLocator**: Fits 2D Gaussian to particle intensity profile
   - Best for: Well-separated circular particles
   - Accuracy: Excellent (< 0.05 pixel RMSE)
   - Speed: Moderate (uses scipy.optimize)

2. **Parabola2DLocator**: Fits parabola through peak for fast refinement
   - Best for: Real-time tracking, well-defined peaks
   - Accuracy: Good (< 0.1 pixel RMSE)
   - Speed: Fast

3. **CrossCorrelationLocator**: Template matching with FFT
   - Best for: Particles with consistent appearance
   - Accuracy: Good
   - Speed: Fast (FFT-based)

#### Propagation Strategies (`src/tracking_toolbox/propagators/`)

1. **DirectPropagator**: Uses previous position as guess
   - Best for: Slow-moving particles, high frame rates

2. **MaxIntensityPropagator**: Searches for maximum intensity near previous position
   - Best for: Faster-moving particles, varying intensity

#### Tracking Workflow (`src/tracking_toolbox/trackers/`)

**IterativeTracker**: Combines locator and propagator for multi-frame tracking
- Refines initial coordinates in frame 0
- Propagates to generate guesses for subsequent frames
- Refines each frame iteratively
- Outputs TrackPy-compatible DataFrame with `frame` column

## Testing

The toolbox includes a comprehensive testing framework with synthetic data generation and analysis tools.

### Run the Main Test Suite

```bash
cd tests
python test_tracking_accuracy.py
```

This generates:
- 200 frames with a particle moving 0.1 pixel/frame
- Tests all three localization algorithms
- Calculates accuracy metrics (RMSE, bias, etc.)
- Displays and saves plots interactively

**Expected Output:**
- RMSE < 0.1 pixels for all methods
- Plots showing trajectory comparison, error evolution, and pixel bias
- Summary statistics printed to console

### Test Results

Results are saved to `tests/test_results/`:
- Trajectory comparison plots
- Error evolution over time
- Pixel bias analysis (TrackPy-style)
- Method comparison bar charts
- Sample frames with overlaid positions

## Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

See `examples/basic_usage.py` for a complete working example.

### Custom Locator Example

```python
from tracking_toolbox.core.base_locator import BaseLocator

class MyCustomLocator(BaseLocator):
    def _refine_single(self, window, guess_x, guess_y):
        # Your custom algorithm here
        # ...
        
        return {
            'x': refined_x,
            'y': refined_y,
            'mass': integrated_intensity,
            'signal': peak_amplitude,
            'size': particle_size,
        }
```

## Documentation

### Directory Structure

```
tracking-toolbox/
├── src/tracking_toolbox/
│   ├── core/              # Base classes
│   ├── locators/          # Localization algorithms
│   ├── propagators/       # Propagation strategies
│   ├── trackers/          # Combined workflows
│   └── utils/             # Utility functions
├── tests/
│   ├── synthetic/         # Synthetic data generation
│   ├── analysis/          # Accuracy metrics and visualization
│   └── test_tracking_accuracy.py
└── examples/
    └── basic_usage.py
```

### Key Features

- **Sub-pixel Accuracy**: All locators achieve < 0.1 pixel RMSE
- **Boundary Handling**: Graceful handling of particles near image edges
- **Noise Robustness**: Tested with Poisson and Gaussian noise
- **Pixel Bias Correction**: Built-in analysis tools for systematic errors
- **TrackPy Compatible**: Direct integration with TrackPy functions

## Performance Tips

1. **Choose the Right Locator**:
   - Gaussian2D: Best accuracy, slower
   - Parabola2D: Fast, good accuracy
   - CrossCorrelation: Fast, good for consistent particle shapes

2. **Window Size**: 
   - Should be odd (default: 11)
   - Should be 4-5x the particle width
   - Larger windows = more robust but slower

3. **Propagation Strategy**:
   - DirectPropagator: Simple, works for slow motion
   - MaxIntensityPropagator: Better for faster particles

## Citation

If you use this toolbox in your research, please cite:

- This toolbox: Tracking Toolbox (https://github.com/andyrward/Tracking_ToolBox)
- TrackPy: Trackpy v0.5.0, DOI: 10.5281/zenodo.3492186

## Acknowledgments

This toolbox is designed to complement [TrackPy](http://soft-matter.github.io/trackpy/), 
an excellent Python particle tracking library by the Soft Matter group.

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

Andy Ward - andyrward@users.noreply.github.com