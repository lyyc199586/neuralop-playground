"""
k_field.py

This script generates a dataset of 2D random conductivity fields k(x,y).
Each k-field is a smooth random scalar field over a grid of specified resolution.

Overview:
- Generates smooth random scalar fields using Gaussian filtering.
- Saves the generated dataset as a NumPy array.

Functions:
- generate_k_binary: Generates a single binary k-field with specified parameters.
- generate_k_dataset: Generates and saves a dataset of binary k-fields.

Author: Yangyuanchen Liu
Date: 2025-04-14
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional

def generate_k_binary(
    grid_shape: Tuple[int, int] = (16, 16),
    smoothness: float = 2.0,
    threshold: float = 0.5,
    k_low: float = 0.1,
    k_high: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a binary conductivity field by thresholding a smoothed random field.

    Parameters:
        grid_shape (Tuple[int, int], optional): Shape of the grid (rows, cols). Defaults to (16, 16).
        smoothness (float, optional): Standard deviation for Gaussian blur; larger values result in smoother regions. Defaults to 2.0.
        threshold (float, optional): Percentile (0~1) to binarize the field. Defaults to 0.5.
        k_low (float, optional): Value assigned to regions below the threshold. Defaults to 0.1.
        k_high (float, optional): Value assigned to regions above the threshold. Defaults to 1.0.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Generated binary conductivity field.
    """
    if seed is not None:
        np.random.seed(seed)
    raw = np.random.rand(*grid_shape)
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(raw, sigma=smoothness)
    norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())  # normalize to [0, 1]
    binary = norm > threshold
    return np.where(binary, k_low, k_high)


def generate_k_dataset(
    n_samples: int = 1000,
    res: Tuple[int, int] = (16, 16),
    seed: int = 42,
    save_path: str = './k_dataset.npy',
) -> None:
    """
    Generates a dataset of 2D random binary conductivity fields and saves it as a NumPy array.

    Parameters:
        n_samples (int, optional): Number of k-fields to generate. Defaults to 1000.
        res (Tuple[int, int], optional): Resolution of each k-field (rows, cols). Defaults to (16, 16).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        save_path (str, optional): Path to save the generated dataset. Defaults to './k_dataset.npy'.

    Returns:
        None
    """
    np.random.seed(seed)
    k_dataset = np.zeros((n_samples, *res))
    for i in range(n_samples):
        k_dataset[i] = generate_k_binary(res, smoothness=2.0, threshold=0.4, k_low=0.1, k_high=1.0, seed=seed + i)
    np.save(save_path, k_dataset)


if __name__ == "__main__":
    # Example usage
    n_samples = 50
    res_x, res_y = 64, 64
    save_path = f"../../data/heat/k_n{n_samples}_res{res_x}.npy"
    generate_k_dataset(n_samples=n_samples, res=(res_x, res_y), save_path=save_path)
