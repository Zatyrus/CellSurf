## Dependencies
import numpy as np

__all__ = ["estimate_magnitude_from_data"]

# %% Utility functions
def estimate_magnitude_from_data(data: np.ndarray) -> float:
    """Estimate the magnitude for a given dataset. This function computes the magnitude based on the range of the data values, providing a heuristic for scaling visualizations or analyses.

    Args:
        data (np.ndarray): The input data for which to estimate the magnitude.
    Returns:
        float: The estimated magnitude based on the data range.
    """
    
    data_range = np.ptp(data)  # Peak-to-peak (max - min) of the data
    magnitude = data_range / 10.0  # Heuristic: scale the range down by a factor of 10
    magnitude = np.floor(np.log10(magnitude))  # Logarithmic scaling of the magnitude
    return magnitude