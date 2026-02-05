"""Direct propagation: use previous position as next guess."""

import numpy as np
from ..core.base_propagator import BasePropagator


class DirectPropagator(BasePropagator):
    """Use previous frame positions as guess coordinates.
    
    This is the simplest propagation strategy: assumes particles don't
    move significantly between frames, so the previous position is used
    directly as the guess for the next frame.
    
    Good for slow-moving particles or high frame rates.
    """
    
    def propagate(self, current_frame, previous_positions, frame_number):
        """Use previous positions as guesses.
        
        Parameters
        ----------
        current_frame : ndarray
            2D image of the current frame (not used by this propagator).
        previous_positions : pandas.DataFrame or array-like
            Positions from the previous frame.
        frame_number : int
            Current frame number (not used by this propagator).
            
        Returns
        -------
        ndarray
            Nx2 array of (x, y) guess coordinates.
        """
        return self._positions_to_array(previous_positions)
