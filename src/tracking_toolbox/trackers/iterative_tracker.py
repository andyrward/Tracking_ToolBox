"""Iterative tracker combining locator and propagator."""

import pandas as pd
import numpy as np


class IterativeTracker:
    """Combine locator and propagator for multi-frame tracking.
    
    This tracker iterates through frames, using a locator to refine positions
    and a propagator to generate guesses for the next frame.
    
    Workflow:
        1. Frame 0: Refine initial guess coordinates using locator
        2. Frame N+1: Use propagator to generate guess from frame N positions
        3. Refine positions with locator
        4. Repeat for all frames
    
    The output is a TrackPy-compatible DataFrame that can be used with
    tp.link() and other TrackPy functions.
    
    Parameters
    ----------
    locator : BaseLocator
        Localization algorithm instance.
    propagator : BasePropagator
        Propagation algorithm instance.
        
    Attributes
    ----------
    locator : BaseLocator
        The locator instance.
    propagator : BasePropagator
        The propagator instance.
    """
    
    def __init__(self, locator, propagator):
        self.locator = locator
        self.propagator = propagator
        
    def track(self, frames, initial_coordinates, start_frame=0):
        """Track particles through multiple frames.
        
        Parameters
        ----------
        frames : array-like
            Sequence of 2D image frames. Can be a 3D array (frame, y, x)
            or a list of 2D arrays.
        initial_coordinates : array-like
            Initial guess coordinates for the first frame. Should be
            a list of (x, y) tuples or Nx2 array.
        start_frame : int, optional
            Frame number to assign to the first frame. Default is 0.
            
        Returns
        -------
        pandas.DataFrame
            TrackPy-compatible DataFrame with columns: frame, x, y, mass,
            signal, size. Each particle gets a row for each frame.
        """
        # Convert frames to list if needed
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            frames_list = [frames[i] for i in range(frames.shape[0])]
        else:
            frames_list = list(frames)
            
        if len(frames_list) == 0:
            raise ValueError("No frames provided")
        
        # Convert initial coordinates to list of tuples
        if isinstance(initial_coordinates, np.ndarray):
            coords = [(x, y) for x, y in initial_coordinates]
        else:
            coords = list(initial_coordinates)
        
        all_results = []
        
        # Process first frame
        result_df = self.locator.locate(frames_list[0], coords)
        if len(result_df) > 0:
            result_df['frame'] = start_frame
            all_results.append(result_df)
            current_positions = result_df
        else:
            # No particles found in first frame
            return pd.DataFrame(columns=['frame', 'x', 'y', 'mass', 'signal', 'size'])
        
        # Process subsequent frames
        for frame_idx in range(1, len(frames_list)):
            frame_number = start_frame + frame_idx
            current_frame = frames_list[frame_idx]
            
            # Generate guess coordinates using propagator
            guess_coords = self.propagator.propagate(
                current_frame,
                current_positions,
                frame_number
            )
            
            # Refine positions using locator
            result_df = self.locator.locate(current_frame, guess_coords)
            
            if len(result_df) > 0:
                result_df['frame'] = frame_number
                all_results.append(result_df)
                current_positions = result_df
            else:
                # No particles found, can't continue
                break
        
        if not all_results:
            return pd.DataFrame(columns=['frame', 'x', 'y', 'mass', 'signal', 'size'])
        
        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns to put frame first
        cols = ['frame', 'x', 'y', 'mass', 'signal', 'size']
        # Include any additional columns that might be present
        for col in final_df.columns:
            if col not in cols:
                cols.append(col)
        
        return final_df[cols]
