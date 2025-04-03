"""
mask.py

Utility functions for mask generation
used in neural operator models for virtual damage sensor.

Author: Yangyuanchen Liu
Date: 2025-03-31
"""

import random
import torch
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt

class MaskGenerator:
    def __init__(self, grid_shape: Tuple[int, int]):
        """
        Initialize the MaskGenerator with the grid shape.

        Args:
            grid_shape (Tuple[int, int]): The shape of the grid (height, width).
        """
        self.grid_shape = grid_shape # (height, width)
        self.masks = None # Torch.Tensor: (n_masks, height, width)


    def generate_from_positions(
        self,
        positions_list: List[List[Tuple[int, int]]],
    ) -> torch.Tensor:
        """
        Generate masks from a list of positions.
        Args:
            positions_list (List[List[Tuple[int, int]]]): A list of lists of positions.
                Each inner list contains tuples representing (x, y) coordinates.
        """
        
        height, width = self.grid_shape
        mask_list = []
        
        for positions in positions_list:
            # Create a mask initialized to zeros
            mask = np.zeros((height, width), dtype=np.float32)
            
            # Set the positions to 1 in the mask
            for (i, j) in positions:
                assert 0 <= i < height and 0 <= j < width, f"Position ({i},{j}) out of bounds for shape ({height},{width})"
                mask[i, j] = 1.0

            mask_list.append(mask)
        self.masks = torch.tensor(mask_list, dtype=torch.float32) # (n_masks, height, width)
        return self.masks 
    
    def generate_random(
        self, n_masks:int, n_sensors_range: Tuple[int, int], top_position: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate a list of sensor layouts with one fixed top sensor and variable non-adjacent right sensors.

        Args:
            n_masks (int): Number of sensor layouts to generate.
            n_sensors_range (Tuple[int, int]): Range of total sensor numbers, e.g., (5, 17).
            top_position (Tuple[int, int]): Fixed sensor position on the top boundary. e.g., (31, 16).
             
        Returns:
            List[List[Tuple[int, int]]]: List of sensor position lists.
        """
        height, width = self.grid_shape
        right_col = width - 1
        right_candidates = [(i, right_col) for i in range(0, height, 2)]
        mask_list = []
        
        for _ in range(n_masks):
            total_sensors = random.randint(n_sensors_range[0], n_sensors_range[1])
            num_right = total_sensors - 1  # one reserved for top

            if num_right > len(right_candidates):
                raise ValueError(f"Cannot place {num_right} non-adjacent sensors on right boundary.")

            selected_right = random.sample(right_candidates, num_right)
            positions  = [top_position] + selected_right
            
            # build mask
            mask = np.zeros((height, width), dtype=np.float32)
            for (i, j) in positions:
                assert 0 <= i < height and 0 <= j < width, f"Position ({i},{j}) out of bounds for shape ({height},{width})"
                mask[i, j] = 1.0
            mask_list.append(mask)

        self.masks = torch.tensor(mask_list, dtype=torch.float32)
        return self.masks
    
    def save(self, save_path: Union[str, Path]):
        """
        Save the generated masks to a file.

        Args:
            save_path (Union[str, Path]): The path to save the masks.
        """
        if self.masks is None:
            raise ValueError("No masks to save. Please generate masks first.")
        
        # Save the masks using torch.save
        torch.save(self.masks, save_path)
        
    def plot(self, idx:int = 0):
        """
        Plot the generated masks using matplotlib.

        Args:
            idx (int): The index of the mask to plot. Default is 0.
        """
        if self.masks is None:
            raise ValueError("No masks to plot. Please generate masks first.")
        
        if idx < 0 or idx >= self.masks.shape[0]:
            raise IndexError(f"Index {idx} out of range for masks with shape {self.masks.shape}.")
        
        plt.imshow(self.masks[idx].numpy(), cmap='gray_r', origin='lower')
        plt.title(f"Mask {idx}")
        plt.show()