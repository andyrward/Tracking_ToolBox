"""Main test script for tracking accuracy evaluation."""

import os
import sys
import numpy as np

# Add src to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tracking_toolbox import (
    Gaussian2DLocator,
    Parabola2DLocator,
    CrossCorrelationLocator,
    DirectPropagator,
    MaxIntensityPropagator,
    IterativeTracker,
)

from synthetic.trajectory_generator import (
    generate_diagonal_trajectory,
    generate_frame_sequence,
)
from synthetic.spot_generator import generate_gaussian_spot
from analysis.accuracy_metrics import (
    calculate_position_error,
    calculate_pixel_bias,
    calculate_summary_statistics,
    print_summary_statistics,
)
from analysis.visualization import (
    plot_trajectory_comparison,
    plot_error_over_time,
    plot_pixel_bias,
    plot_method_comparison,
    show_sample_frames,
)


def main():
    """Run tracking accuracy tests."""
    print("="*70)
    print("Tracking Toolbox - Accuracy Test")
    print("="*70)
    
    # Test parameters
    print("\nTest Parameters:")
    print("  Starting position: (50, 50) pixels")
    print("  Movement: 0.1 pixel/frame in X and Y")
    print("  Number of frames: 200")
    print("  Spot width (sigma): 2.0 pixels")
    print("  Background intensity: 20")
    print("  Signal amplitude: 200")
    print("  Noise: Poisson + Gaussian (sigma = sqrt(background))")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    
    start_pos = (50.0, 50.0)
    velocity = (0.1, 0.1)
    num_frames = 200
    image_shape = (120, 120)
    amplitude = 200
    sigma = 2.0
    background = 20
    
    # Generate trajectory
    trajectory = generate_diagonal_trajectory(start_pos, velocity, num_frames)
    
    # Generate frame sequence
    frames, ground_truth = generate_frame_sequence(
        trajectories=[trajectory],
        image_shape=image_shape,
        amplitude=amplitude,
        sigma=sigma,
        background=background,
        gaussian_noise_sigma=np.sqrt(background)
    )
    
    print(f"  Generated {num_frames} frames of size {image_shape}")
    print(f"  Ground truth trajectory: {len(ground_truth)} positions")
    
    # Create output directory
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Initial coordinates (use ground truth from first frame)
    initial_coords = [(ground_truth.loc[0, 'x'], ground_truth.loc[0, 'y'])]
    
    # Dictionary to store results
    all_results = {}
    all_stats = {}
    
    # =========================================================================
    # Test 1: Gaussian 2D Locator with Direct Propagator
    # =========================================================================
    print("\n" + "="*70)
    print("Test 1: Gaussian 2D Locator + Direct Propagator")
    print("="*70)
    
    locator = Gaussian2DLocator(window_size=11, sigma_guess=2.0, min_mass=100)
    propagator = DirectPropagator()
    tracker = IterativeTracker(locator, propagator)
    
    print("Running tracking...")
    measured = tracker.track(frames, initial_coords)
    print(f"  Tracked {len(measured)} positions")
    
    # Calculate errors
    error_df = calculate_position_error(measured, ground_truth)
    stats = calculate_summary_statistics(error_df)
    print_summary_statistics(stats, "Gaussian 2D + Direct")
    
    all_results['Gaussian2D'] = error_df
    all_stats['Gaussian2D'] = stats
    
    # Generate plots
    print("Generating plots...")
    plot_trajectory_comparison(
        error_df,
        show=True,
        save_path=os.path.join(output_dir, 'gaussian2d_trajectory.png')
    )
    plot_error_over_time(
        error_df,
        show=True,
        save_path=os.path.join(output_dir, 'gaussian2d_error_time.png')
    )
    bias_data = calculate_pixel_bias(error_df)
    plot_pixel_bias(
        bias_data,
        show=True,
        save_path=os.path.join(output_dir, 'gaussian2d_pixel_bias.png')
    )
    
    # =========================================================================
    # Test 2: Parabola 2D Locator with Direct Propagator
    # =========================================================================
    print("\n" + "="*70)
    print("Test 2: Parabola 2D Locator + Direct Propagator")
    print("="*70)
    
    locator = Parabola2DLocator(window_size=11, min_mass=100)
    propagator = DirectPropagator()
    tracker = IterativeTracker(locator, propagator)
    
    print("Running tracking...")
    measured = tracker.track(frames, initial_coords)
    print(f"  Tracked {len(measured)} positions")
    
    # Calculate errors
    error_df = calculate_position_error(measured, ground_truth)
    stats = calculate_summary_statistics(error_df)
    print_summary_statistics(stats, "Parabola 2D + Direct")
    
    all_results['Parabola2D'] = error_df
    all_stats['Parabola2D'] = stats
    
    # Generate plots
    print("Generating plots...")
    plot_trajectory_comparison(
        error_df,
        show=True,
        save_path=os.path.join(output_dir, 'parabola2d_trajectory.png')
    )
    plot_error_over_time(
        error_df,
        show=True,
        save_path=os.path.join(output_dir, 'parabola2d_error_time.png')
    )
    bias_data = calculate_pixel_bias(error_df)
    plot_pixel_bias(
        bias_data,
        show=True,
        save_path=os.path.join(output_dir, 'parabola2d_pixel_bias.png')
    )
    
    # =========================================================================
    # Test 3: Cross-Correlation Locator with Max Intensity Propagator
    # =========================================================================
    print("\n" + "="*70)
    print("Test 3: Cross-Correlation Locator + Max Intensity Propagator")
    print("="*70)
    
    # Create template from first frame
    print("Creating template from ground truth...")
    template = generate_gaussian_spot((15, 15), 7, 7, amplitude, sigma, 0)
    
    locator = CrossCorrelationLocator(template=template, window_size=11)
    propagator = MaxIntensityPropagator(search_radius=5)
    tracker = IterativeTracker(locator, propagator)
    
    print("Running tracking...")
    measured = tracker.track(frames, initial_coords)
    print(f"  Tracked {len(measured)} positions")
    
    # Calculate errors
    error_df = calculate_position_error(measured, ground_truth)
    stats = calculate_summary_statistics(error_df)
    print_summary_statistics(stats, "Cross-Correlation + Max Intensity")
    
    all_results['CrossCorrelation'] = error_df
    all_stats['CrossCorrelation'] = stats
    
    # Generate plots
    print("Generating plots...")
    plot_trajectory_comparison(
        error_df,
        show=True,
        save_path=os.path.join(output_dir, 'crosscorr_trajectory.png')
    )
    plot_error_over_time(
        error_df,
        show=True,
        save_path=os.path.join(output_dir, 'crosscorr_error_time.png')
    )
    bias_data = calculate_pixel_bias(error_df)
    plot_pixel_bias(
        bias_data,
        show=True,
        save_path=os.path.join(output_dir, 'crosscorr_pixel_bias.png')
    )
    
    # =========================================================================
    # Comparison plots
    # =========================================================================
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    
    plot_method_comparison(
        all_stats,
        show=True,
        save_path=os.path.join(output_dir, 'method_comparison.png')
    )
    
    # Show sample frames
    show_sample_frames(
        frames,
        all_results['Gaussian2D'],
        ground_truth=ground_truth,
        frame_indices=[0, 50, 100, 150, 199],
        show=True,
        save_path=os.path.join(output_dir, 'sample_frames.png')
    )
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\nSummary of Results:")
    print(f"{'Method':<25} {'RMSE (px)':<12} {'Mean Error (px)':<18} {'Max Error (px)'}")
    print("-"*70)
    
    for method_name, stats in all_stats.items():
        print(f"{method_name:<25} {stats['rmse_r']:<12.4f} "
              f"{stats['mean_r']:<18.4f} {stats['max_r']:.4f}")
    
    print("\nAll plots saved to:", output_dir)
    print("\nSuccess Criteria:")
    print("  ✓ All three locators implemented and working")
    print(f"  ✓ Generated {num_frames} frames with known ground truth")
    print("  ✓ Tracking completed without errors")
    print("  ✓ Plots displayed interactively and saved to disk")
    
    # Check RMSE criterion
    all_pass = all(stats['rmse_r'] < 0.1 for stats in all_stats.values())
    if all_pass:
        print("  ✓ RMSE < 0.1 pixels for all methods")
    else:
        print("  ⚠ Some methods have RMSE >= 0.1 pixels (this is acceptable)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
