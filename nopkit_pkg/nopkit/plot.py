#%%
import torch
import matplotlib.pyplot as plt

# data_path = '/home/yliu664/scr4_sghosh20/yang/no_playground/neuralop-playground/data/damage_sensor/ramp_n500_res32.pt'
# data_path = '/home/yliu664/scr4_sghosh20/yang/no_playground/neuralop-playground/data/damage_sensor/damage_n500_t175_res32.pt'
# data_path = '/home/yliu664/scr4_sghosh20/yang/no_playground/neuralop-playground/data/damage_sensor/defgrad_n500_t175_res32.pt'
data_path = '/home/yliu664/scr4_sghosh20/yang/no_playground/neuralop-playground/data/damage_sensor/elec_n500_t175_res32.pt'
data = torch.load(data_path)

print(f"Loaded data shape: {data.shape}")

# plt.imshow(data[0,:,:], origin='lower')
# plt.contourf(data[0,:,:,-1], levels=50, origin='lower')
plt.imshow(data[0,:,:,-1], origin='lower', cmap='rainbow')
# %%
