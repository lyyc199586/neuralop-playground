#%%
from pathlib import Path
from neuralop.models import FNO
from nopkit.data import load_damage_sensor_dataset
from nopkit.plot import *

device = 'cuda'

data_dir = Path("../data/damage_sensor")

train_loader, test_loaders, data_processor = load_damage_sensor_dataset(
    ramps_path=data_dir / "ramps_n500_res32.pt",
    damage_path=data_dir / "damage_n500_t175_res32.pt",
    defgrad_path=data_dir / "defgrad_n500_t175_res32.pt",
    elec_path=data_dir / "elec_n500_t175_res32.pt",
    masks_path=data_dir / "mask_saikat.pt",
    n_train=450,
    batch_size=10,
    test_batch_sizes=[10],
    test_resolutions=[32],
    encode_input=True,
    encode_output=True,
    # encoding="channel-wise",
    # channel_dim=1,
)
data_processor = data_processor.to(device)

model_reload = FNO.from_checkpoint("./model", save_name="fno3d_cuda")

model_use = model_reload.to(device)

test_samples = test_loaders[32].dataset
idx = 10

data = test_samples[idx]

data_processor.to(device)
data_processor.eval()
data = data_processor.preprocess(data, batched=True)


x = data['x']
y = data['y']
out = model_use(x)

# postprocess data
out, _ = data_processor.postprocess(out, data)

t_range = range(150, 175)

vmins = [1.5e6, 1.005, 0.1]
vmaxs = [7.5e6, 1.04, 0.6]

save_path = f'./media/mask16_id{idx}_3d.gif'
pred_voxel_anim(y.cpu(), out.squeeze().detach().cpu(), t_range=t_range,
          save_path=save_path, fps=2,
          show_colorbar=True, vmins=vmins, vmaxs=vmaxs, plot_method='contourf')
# %%
