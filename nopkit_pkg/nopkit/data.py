"""
data.py

Utility functions for loading and preprocessing tensor datasets
used in neural operator models for virtual damage sensor.

Author: Yangyuanchen Liu
Date: 2025-03-28

Overview:
- Loads RAMPs and spatiotemporal variable `.npy` files from FEM results directories.
- Stacks and reshapes data into `torch.Tensor` formats.
- Saves preprocessed data as `.pt` files with structured naming.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Union
from einops import rearrange
from torch.utils.data import DataLoader
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

def get_ramps_files(path: Path) -> List[Path]:
    """
    Retrieves sorted file paths to RAMP `.npy` files from InputsSpatial directory.

    Args:
        path (Path): Root directory containing 'InputsSpatial'.

    Returns:
        List[Path]: Sorted list of RAMP file paths.
    """
    
    # each volume fraction data have shape (res_x, res_y)
    ramp_path = path / 'InputsSpatial'
    # sort macro dir
    macro_dirs = sorted([
      d for d in ramp_path.iterdir()
      if d.is_dir() and d.name.startswith('Macrostructure')
    ], key=lambda p: int(''.join(filter(str.isdigit, p.name))))
    
    ramps_files = []
    for d in macro_dirs:
        ramp_file = d / 'output' / 'volume fraction_slice1.npy'
        if ramp_file.exists():
            ramps_files.append(ramp_file)
        
    return ramps_files
    
def get_var_files(path: Path, variable: str) -> List[Path]:
    """
    Retrieves sorted file paths for a given variable from InputsSpatioTemporal directory.

    Args:
        path (Path): Root directory containing 'InputsSpatioTemporal'.
        variable (str): Variable name (e.g., 'damage3', 'deformation gradient2').

    Returns:
        List[Path]: Sorted list of variable file paths.
    """
    
    # each variable data have shape (time_steps, res_x, res_y)
    var_path = path / 'InputsSpatioTemporal'
    macro_dirs = sorted([
        d for d in var_path.iterdir()
        if d.is_dir() and d.name.startswith('Macrostructure')
    ], key=lambda p: int(''.join(filter(str.isdigit, p.name))))
    
    var_files = []
    for d in macro_dirs:
        var_file = d / 'output' / f'{variable}_slice1.npy'
        if var_file.exists():
            var_files.append(var_file)
            
    return var_files

def prepare_dataset(load_dir:Path, save_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Loads and preprocesses RAMPs and spatiotemporal variables from the given directory.

    Args:
        load_dir (Path): Root directory containing 'InputsSpatial' and 'InputsSpatioTemporal'.
        save_dir (Path): Target directory to save processed `.pt` tensors.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing preprocessed tensors:
            - 'ramps': shape (n_samples, res_x, res_y)
            - 'damage': shape (n_samples, res_x, res_y, time_steps)
            - 'defgrad': shape (n_samples, res_x, res_y, time_steps)
            - 'elec': shape (n_samples, res_x, res_y, time_steps)
    """
    ramps_files = get_ramps_files(load_dir)
    damage_files = get_var_files(load_dir, 'damage3')
    defgrad_files = get_var_files(load_dir, 'deformation gradient2')
    elec_files = get_var_files(load_dir, 'electric field3')
    
    # load RAMPs: (n_samples, res_x, res_y)
    ramps_list = []
    for i, f in enumerate(ramps_files):
        print(f"[ramp] Loading {i+1}/{len(ramps_files)}: {f.name}")
        ramps_list.append(torch.tensor(np.load(f), dtype=torch.float32))
    ramps_data = torch.stack(ramps_list)
    n_samples, res_x, res_y = ramps_data.shape
    ramps_filename = f"ramps_n{n_samples}_res{res_x}.pt"
    
    # helper function to load variables
    def load_var(files: List[Path], var_name: str) -> Tuple[torch.Tensor, str]:
        """
        Loads and rearranges variable `.npy` files into a 4D tensor.
        
        Args:
            files (List[Path]): List of file paths to load.
            var_name (str): Variable name used for saving the output filename.

        Returns:
            Tuple[torch.Tensor, str]: 
                - Tensor of shape (n_samples, res_x, res_y, time_steps)
                - Suggested output filename for saving
        """
        var_list = []
        for i, f in enumerate(files):
            print(f"[{var_name}] Loading {i+1}/{len(files)}: {f.name}")
            data = torch.tensor(np.load(f), dtype=torch.float32) # (time_steps, res_x, res_y)
            data = rearrange(data, 't x y -> x y t') # (res_x, res_y, time_steps)
            var_list.append(data)
        var_stack = torch.stack(var_list) # (n_samples, res_x, res_y, time_steps)
        n_samples, res_x, res_y, time_steps = var_stack.shape
        filename = f"{var_name}_n{n_samples}_t{time_steps}_res{res_x}.pt"
        return var_stack, filename
    
    damage_data, damage_filename = load_var(damage_files, "damage")
    defgrad_data, defgrad_filename = load_var(defgrad_files, "defgrad")
    elec_data, elec_filename = load_var(elec_files, "elec")

    # Save dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    torch.save(ramps_data, save_dir/ramps_filename)
    torch.save(damage_data, save_dir/damage_filename)
    torch.save(defgrad_data, save_dir/defgrad_filename)
    torch.save(elec_data, save_dir/elec_filename)
    
    # Print info
    print(f"Saved RAMPs      -> {save_dir/ramps_filename}, ({ramps_data.shape})")
    print(f"Saved damage     -> {save_dir/damage_filename}, ({damage_data.shape})")
    print(f"Saved defgrad    -> {save_dir/defgrad_filename}, ({defgrad_data.shape})")
    print(f"Saved elec field -> {save_dir/elec_filename}, ({elec_data.shape})")

    return {
        'ramps': ramps_data,
        'damage': damage_data,
        'defgrad': defgrad_data,
        'elec': elec_data,
    }
    

class DamageSensorDataset:
    """
    This class is based on neuralop/data/datasets/pt_dataset.py,
    1. split data into train and test,
    2. include the masks for electric field to generate sparse field as input
        x: [Ek(x, t), M(x), RAMPs(x)], where Ek(x, t) = E(x, t)@M(x), M and RAMPs are extended to match timesteps
        y: [E(x, t), F(x, t), D(x, t)]
    3. preprocess data (normalization)
    we should add the ability to extend multiple masks (generated randomly) in general (provide a mask list)
    
    Just like PTdataset, this dataset are required to expose the following attributes after init:
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """
    def __init__(self,
                 ramps_path: Union[Path, str],
                 damage_path: Union[Path, str],
                 defgrad_path: Union[Path, str],
                 elec_path: Union[Path, str],
                 masks_path: Union[Path, str],
                 n_train:int,
                 batch_size: int,
                 test_batch_sizes: List[int],
                 test_resolutions: int=[32],
                 encode_input: bool=True,
                 encode_output: bool=True,
                 encoding: str = "channel-wise",
                 channel_dim: int = 1 # 
                 ):
        """
        channel_dim : int, optional
            dimension of saved tensors to index data channels, 
            by default 1 (i.e, n_samples, n_channels, dim_1, dim_2, ...)
        """

        #### Load data
        ramps = torch.load(ramps_path)      # shape (n_samples, res_x, res_y)
        damage = torch.load(damage_path)    # shape (n_samples, res_x, res_y, time_steps)
        defgrad = torch.load(defgrad_path)  # shape (n_samples, res_x, res_y, time_steps)
        elec = torch.load(elec_path)        # shape (n_samples, res_x, res_y, time_steps)
        masks = torch.load(masks_path)      # shape (n_masks,   res_x, res_y)
        
        # print info
        print(f"Loaded RAMPs      -> {ramps_path},\t({ramps.shape})")
        print(f"Loaded damage     -> {damage_path},\t({damage.shape})")
        print(f"Loaded defgrad    -> {defgrad_path},\t({defgrad.shape})")
        print(f"Loaded elec field -> {elec_path},\t({elec.shape})")
        print(f"Loaded masks      -> {masks_path},\t({masks.shape})")
        
        # shape check
        assert ramps.ndim == 3, "RAMPs should have shape (n_samples, res_x, res_y)"
        assert damage.ndim == 4 and defgrad.ndim == 4 and elec.ndim == 4, "Variables should have shape (n_samples, res_x, res_y, time_steps)"
        assert ramps.shape[0] == damage.shape[0] == defgrad.shape[0] == elec.shape[0], "Sample counts must match"
        assert ramps.shape[1:] == damage.shape[1:3] == defgrad.shape[1:3] == elec.shape[1:3], "Spatial shapes must match"
        assert damage.shape[3] == defgrad.shape[3] == elec.shape[3], "Time steps must match across all variables"

        
        #### Prepare data
        inputs = []
        outputs = []
        n_samples, res_x, res_y, time_steps = damage.shape
        n_masks = masks.shape[0]
        
        for i in range(n_samples):
            ramp = ramps[i]     # (res_x, res_y)
            e = elec[i]         # (res_x, res_y, time_steps)
            f = defgrad[i]      # (res_x, res_y, time_steps)
            d = damage[i]       # (res_x, res_y, time_steps)
            
            for mask in masks:
                # apply mask to electric field， Ek = E*M
                e_masked = e * mask.unsqueeze(-1)  # Broadcasting to apply mask, shape (res_x, res_y, time_steps)
                
                # expand mask and ramp to match time steps
                mask_expanded = mask.unsqueeze(-1).expand(res_x, res_y, time_steps)
                ramp_expanded = ramp.unsqueeze(-1).expand(res_x, res_y, time_steps)
                
                # stack inputs: (n_channels=3, res_x, res_y, time_steps)
                x = torch.stack([e_masked, mask_expanded, ramp_expanded], dim=0)
                y = torch.stack([e, f, d], dim=0)
                
                inputs.append(x)
                outputs.append(y)
                
        x_all = torch.stack(inputs)  # (n_masks*n_samples, n_channels, res_x, res_y, time_steps)
        y_all = torch.stack(outputs) # (n_masks*n_samples, n_channels, res_x, res_y, time_steps)
            
        x_train = x_all[:n_train]
        y_train = y_all[:n_train]
        
        x_test = x_all[n_train:]
        y_test = y_all[n_train:]
        
        # print train and test info
        print(f"Loading total samples: {n_samples}, total masks: {n_masks}")
        print(f"Loading train db: {x_train.shape[0]} samples, test db: {x_test.shape[0]} samples")
        print(f"Train input shape: {x_train.shape}, Train output shape: {y_train.shape}")
        print(f"Test input shape: {x_test.shape}, Test output shape: {y_test.shape}")
        
        #### Normalization of data
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                # preseve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]
                
            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
        else:
            input_encoder = None
            
        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None
            
        #### Create train db, dataprocessor and test db
        self._train_db = TensorDataset(
            x_train,
            y_train
        )
        
        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder)
        
        self._test_dbs = {} # we only consider 1 test db with the same resolution for now
        for res in test_resolutions:
            
            test_db = TensorDataset(
                x_test,
                y_test
            )
            self._test_dbs[res] = test_db
        
        del ramps, damage, defgrad, elec, masks
        
    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_dbs(self):
        return self._test_dbs

def load_damage_sensor_dataset(
    ramps_path: Union[Path, str],
    damage_path: Union[Path, str],
    defgrad_path: Union[Path, str],
    elec_path: Union[Path, str],
    masks_path: Union[Path, str],
    n_train: int,
    batch_size: int,
    test_batch_sizes: List[int],
    test_resolutions: int=[32],
    encode_input: bool = True,
    encode_output: bool = False,
    encoding: str = "channel-wise",
    channel_dim: int = 1
) -> DamageSensorDataset:
    """
    Load and prepare the Damage Sensor dataset.

    Args:
        ramps_path (Union[Path, str]): Path to RAMPs tensor file.
        damage_path (Union[Path, str]): Path to damage tensor file.
        defgrad_path (Union[Path, str]): Path to deformation gradient tensor file.
        elec_path (Union[Path, str]): Path to electric field tensor file.
        masks_path (Union[Path, str]): Path to masks tensor file.
        n_train (int): Number of training samples (should be less than n_samples).
        test_resolution (int): Spatial resolution for test dataset.
        encode_input (bool): Whether to encode input data.
        encode_output (bool): Whether to encode output data.
        encoding (str): Encoding method ('channel-wise' or 'pixel-wise').
        channel_dim (int): Dimension for indexing data channels.

    Returns:
        DamageSensorDataset: The loaded dataset object.
    """
    
    dataset = DamageSensorDataset(
        ramps_path=ramps_path,
        damage_path=damage_path,
        defgrad_path=defgrad_path,
        elec_path=elec_path,
        masks_path=masks_path,
        n_train=n_train,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim
    )
    
    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)
    
    test_loaders = {}
    for res,test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       persistent_workers=False,)
    
    return train_loader, test_loaders, dataset.data_processor
        
# testing

if __name__ == "__main__":
    data_path = Path('/home/yliu664/scr4_sghosh20/yang/damagesensor/4_ExtractInputsConvLSTM')
    save_path = Path('/home/yliu664/scr4_sghosh20/yang/no_playground/neuralop-playground/data/damage_sensor')

    dataset = prepare_dataset(data_path, save_path)

    for key, value in dataset.items():
        print(f"{key}: {value.shape}")

