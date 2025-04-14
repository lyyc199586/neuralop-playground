"""
k_field.py

This script generates a dataset of 2D random conductivity fields k(x,y).
Each k-field is a smooth random scalar field over a grid of specified resolution.

Overview:
- Generates smooth random scalar fields using Gaussian filtering.
- Saves the generated dataset as a NumPy array.

Functions:
- generate_k_field: Generates a single k-field with specified parameters.
- generate_k_dataset: Generates and saves a dataset of k-fields.

Author: Yangyuanchen Liu
Date: 2025-04-14
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple


def generate_k_field(
    grid_shape: Tuple[int, int] = (16, 16),
    smooth: float = 1.5,
    k_min: float = 0.1,
    k_max: float = 1.0,
) -> np.ndarray:
    """
    Generates a single 2D random conductivity field (k-field).

    Args:
        grid_shape (Tuple[int, int], optional): Shape of the grid (res_x, res_y). Defaults to (16, 16).
        smooth (float, optional): Standard deviation for Gaussian smoothing. Defaults to 1.5.
        k_min (float, optional): Minimum value of the conductivity field. Defaults to 0.1.
        k_max (float, optional): Maximum value of the conductivity field. Defaults to 1.0.

    Returns:
        np.ndarray: A 2D array representing the k-field.
    """
    noise = np.random.rand(*grid_shape)
    k = gaussian_filter(noise, sigma=smooth)
    k = (k - k.min()) / (k.max() - k.min()) # normalize to [0, 1]
    return k_min + k * (k_max - k_min)


def generate_k_dataset(
    n_samples: int = 1000,
    res: Tuple[int, int] = (16, 16),
    seed: int = 42,
    save_path: str = './k_dataset.npy',
) -> None:
    """
    Generates a dataset of 2D random conductivity fields and saves it as a NumPy array.

    Args:
        n_samples (int, optional): Number of k-fields to generate. Defaults to 1000.
        res (Tuple[int, int], optional): Resolution of each k-field (res_x, res_y). Defaults to (16, 16).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        save_path (str, optional): Path to save the generated dataset. Defaults to './k_dataset.npy'.

    Returns:
        None
    """
    np.random.seed(seed)
    k_dataset = np.zeros((n_samples, *res))
    for i in range(n_samples):
        k_dataset[i] = generate_k_field(res)
    np.save(save_path, k_dataset)


n_samples = 1000
res_x, res_y = 16, 16
save_path = f"../../data/heat/k_n{n_samples}_res{res_x}.npy"
generate_k_dataset(n_samples=n_samples, res=(res_x, res_y), save_path=save_path)
