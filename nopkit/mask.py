"""
mask.py

Utility functions for mask generation
used in neural operator models for virtual damage sensor.

Author: Yangyuanchen Liu
Date: 2025-03-31
"""
#%%
import numpy as np
import torch
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
        
        plt.imshow(self.masks[idx].numpy(), cmap='gray')
        plt.title(f"Mask {idx}")
        plt.show()
    
    
#%% testing
if __name__ == "__main__":
    gen = MaskGenerator(grid_shape=(32, 32))
    positions_list = [
        [(0, 0), (1, 1), (2, 2)],  # Mask 1
        [(10, 10), (11, 11)],     # Mask 2
        [(20, 20), (21, 21), (22, 22)] # Mask 3
    ]
    masks = gen.generate_from_positions(positions_list)
    gen.plot(idx=0)  # Plot the first mask
    gen.save("test.pt")
    
# %%
