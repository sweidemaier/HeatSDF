import os
import sys
import numpy as np
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
from trainers.utils.diff_ops import gradient
from trainers.utils.igp_utils import sample_points
import matplotlib.pyplot as plt
import torch
from trainers.utils.new_utils import tens
import evaluation_surfaces
import trimesh
new_net_path = "/home/weidemaier/PDE Net/NFGP/logs/create_neural_fields_2024-Nov-14-14-41-37"
import os.path as osp
from utils import load_imf
net, cfg = load_imf(
    new_net_path, 
    return_cfg=False)
bs = 100000
from trainers import analyticSDFs
from trainers.utils.vis_utils import imf2mesh

mesh = imf2mesh(net, res = 128, threshold = 0)
trimesh.exchange.export.export_mesh(mesh, "exported.obj", file_type=None, resolver=None)
xy = evaluation_surfaces.torus(bs)


xy_tens = tens(xy)
vec = net(xy_tens)
A = vec.detach().cpu().numpy()
print(A)
val = A + xy


np.savetxt("neural_deform.csv",  val, delimiter = ",", header = "x,y,z")
np.savetxt("reference_config.csv",  xy, delimiter = ",", header = "x,y,z")