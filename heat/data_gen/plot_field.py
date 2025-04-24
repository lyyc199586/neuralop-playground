#%% plot
import matplotlib.pyplot as plt
import numpy as np

n_samples = 1000
res_x, res_y = 32, 32
save_path = f"../../data/heat/k_n{n_samples}_res{res_x}.npy"
k_fields = np.load(save_path)
plt.figure(figsize=(10, 5))

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(k_fields[i], cmap='gray_r', vmin=0, vmax=1)
    # plt.imshow(k_fields[i], cmap='viridis')
    plt.title(f'k-field {i+1}')
    # plt.colorbar()
    plt.axis('off')

# %%
