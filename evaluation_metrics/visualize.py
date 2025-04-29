import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import trimesh
import numpy as np
from trainers.utils.vis_utils import imf2mesh
from utils import load_imf

net_path = "/home/weidemaier/HeatSDF/logs/SDF2025-Apr-23-10-13-10/SDF_step"
### load network
net, cfg = load_imf(
    net_path, 
    return_cfg=False
)
### create mesh and save as .obj
mesh = imf2mesh(net, res = 256, normalize=True, bound = 1.15, threshold=0.0)
trimesh.exchange.export.export_mesh(mesh, net_path + "/vis" + ".obj", file_type=None, resolver=None) 
