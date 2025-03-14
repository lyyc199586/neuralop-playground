#%% check shape
# darcy_test_16.pt: x, y: 50, 16, 16
# darcy_test_32.pt: x, y: 50, 32, 32
# darcy_train_16.pt: x, y: 1000, 16, 16


import torch

data = torch.load("darcy_train_16.pt")
print(type(data))
print(data['x'].shape)
print(data['y'].shape)
# %%
