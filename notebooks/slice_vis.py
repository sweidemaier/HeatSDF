import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import torch
import trimesh
from trainers.utils.new_utils import tens
from trainers.utils.vis_utils import imf2mesh
from trainers import RHS, analyticSDFs
from utils import load_imf, write_obj
from trainers.utils.diff_ops import gradient
import numpy as np
import pyevtk

sample_size = 100
path = "/home/weidemaier/PDE Net/logs/NeuralSDFs_2025-Mar-12-11-15-47"
net, cfg = load_imf(
        path, 
        return_cfg=False
    #,ckpt_fpath = new_net_path + "/checkpoints/epoch_1_iters_2000.pt"
        )
linsp = np.linspace(-1,1,sample_size)
xyz = [None]*(sample_size**2)
for i in range(sample_size):
    for j in range(sample_size):
        xyz[sample_size*i + j] = [linsp[i], linsp[j], 0]

xyz = tens(xyz)
eikonal = (torch.norm(gradient(net(xyz), xyz), dim = -1) - torch.ones_like(torch.norm(gradient(net(xyz), xyz), dim = -1)))
print(eikonal.shape)
eikonal = eikonal.detach().cpu().numpy()
xyz = xyz.detach().cpu().numpy()
vec = [None]*(sample_size**2)
for k in range(len(eikonal)):
    vec[k] = [xyz[k, 0], xyz[k,1], eikonal[k]]
np.savetxt("tests%d.csv", vec , delimiter = ",", header = "x,y,z")

res = 128
VOLUME_PLOT_RESOLUTION = res
F, cfg = load_imf(
    path, 
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
def eikonal(x):
    return (torch.norm(gradient(net(x), x), dim = -1) - torch.ones_like(torch.norm(gradient(net(x), x), dim = -1))).detach().cpu()
def grad(x):
    return (torch.norm(gradient(net(x), x), dim = -1) ).detach().cpu()

# region Error plot

value = torch.cat([grad(tens(v)) for v in split_data]).cpu()


pyevtk.hl.gridToVTK(os.path.join(path, f"slice_viz"),
                    coords[0].reshape(VOLUME_PLOT_RESOLUTION, VOLUME_PLOT_RESOLUTION,
                                      VOLUME_PLOT_RESOLUTION),
                    coords[1].reshape(VOLUME_PLOT_RESOLUTION, VOLUME_PLOT_RESOLUTION,
                                      VOLUME_PLOT_RESOLUTION),
                    coords[2].reshape(VOLUME_PLOT_RESOLUTION, VOLUME_PLOT_RESOLUTION,
                                      VOLUME_PLOT_RESOLUTION),
                    pointData={"pf": value.detach().numpy().flatten()})
