import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import pyevtk
from utils import load_imf

new_net_path = "/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs_2025-Mar-16-16-44-15"
res = 128
VOLUME_PLOT_RESOLUTION = res
F, cfg = load_imf(
    new_net_path, 
    return_cfg=False
    #, ckpt_fpath = new_net_path + "/checkpoints/epoch_3_iters_4000.pt"
    )
# region Volume plot setup
X = np.linspace(-1.2, 1.2, res, dtype='float32')
Y = np.linspace(-1.2, 1.2, res, dtype='float32')
Z = np.linspace(-1.2, 1.2, res, dtype='float32')

coords = np.meshgrid(X, Y, Z)

all_data = torch.tensor(np.array(coords).T.reshape(res ** 3, 3))
batch_size = 2048
split_data = torch.split(all_data, batch_size)
# endregion

# region Error plot
with torch.no_grad():
    value = torch.cat([F(v.cuda()) for v in split_data]).cpu()


pyevtk.hl.gridToVTK(os.path.join(new_net_path, f"plotpf"),
                    coords[0].reshape(VOLUME_PLOT_RESOLUTION, VOLUME_PLOT_RESOLUTION,
                                      VOLUME_PLOT_RESOLUTION),
                    coords[1].reshape(VOLUME_PLOT_RESOLUTION, VOLUME_PLOT_RESOLUTION,
                                      VOLUME_PLOT_RESOLUTION),
                    coords[2].reshape(VOLUME_PLOT_RESOLUTION, VOLUME_PLOT_RESOLUTION,
                                      VOLUME_PLOT_RESOLUTION),
                    pointData={"pf": value.detach().numpy().flatten()})
# endregion