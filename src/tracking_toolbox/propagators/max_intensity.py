"""Max intensity propagation: search for maximum intensity near previous position."""

import numpy as np
from ..core.base_propagator import BasePropagator


class MaxIntensityPropagator(BasePropagator):
    """Search for maximum intensity pixel near previous position.
    
    This propagator searches within a radius around the previous position
    and uses the location of the maximum intensity pixel as the guess.
    
    Good for bright particles that may move between frames.
    
    Parameters
    ----------
    search_radius : int, optional
        Radius in pixels to search around previous position. Default is 5.
        
    Attributes
    ----------
    search_radius : int
        Search radius in pixels.
    """
    
    def __init__(self, search_radius=5):
        self.search_radius = search_radius
        
    def propagate(self, current_frame, previous_positions, frame_number):
        """Find maximum intensity near previous positions.
        
        Parameters
        ----------
        current_frame : ndarray
            2D image of the current frame.
        previous_positions : pandas.DataFrame or array-like
            Positions from the previous frame.
        frame_number : int
            Current frame number (not used by this propagator).
            
        Returns
        -------
        list
            List of (x, y) guess coordinates.
        """
        positions = self._positions_to_array(previous_positions)
        guesses = []
        
        h, w = current_frame.shape
        
        for x, y in positions:
            # Define search region
            x_int = int(round(x))
            y_int = int(round(y))
            
            x_min = max(0, x_int - self.search_radius)
            x_max = min(w, x_int + self.search_radius + 1)
            y_min = max(0, y_int - self.search_radius)
            y_max = min(h, y_int + self.search_radius + 1)
            
            # Extract search region
            region = current_frame[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                # Fallback to previous position
                guesses.append((x, y))
                continue
            
            # Find maximum in region
            max_idx = np.unravel_index(np.argmax(region), region.shape)
            max_y_local, max_x_local = max_idx
            
            # Convert to frame coordinates
            guess_x = x_min + max_x_local
            guess_y = y_min + max_y_local
            
            guesses.append((guess_x, guess_y))
        
        return guesses
