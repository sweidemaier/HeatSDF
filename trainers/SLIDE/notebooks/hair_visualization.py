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

bs = 1000
new_net_path = "/home/weidemaier/PDE Net/logs/NeuralSDFs2_2025-Feb-28-10-01-23"
delta = 0.0001
steps = 500


net, cfg = load_imf(
    new_net_path, 
    return_cfg=False
    )

mesh = imf2mesh(net, res = 256, normalize=True, bound = 1.3, threshold=0.0)

verts = tens(trimesh.sample.sample_surface_even(mesh, bs)[0])
hair = verts
last = verts
for step in range(steps):
    grad = gradient(net(last), last)
    vec = last + 10*delta*grad
    last = last + delta*grad
    hair = torch.cat((hair, vec), dim = 0 )

np.savetxt("hair_viz.csv", hair.detach().cpu().numpy() , delimiter = ",", header = "x,y,z")