import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import pyevtk
from utils import load_imf
from trainers.utils.vis_utils import imf2mesh
from trainers.utils.new_utils import tens
from trainers.utils.diff_ops import gradient
import trimesh
import matplotlib.pyplot as plt

new_net_path = "/home/weidemaier/PDE Net/logs/new looped heads/NeuralSDFs2_2025-Feb-28-00-49-20"
bs = 1

net, cfg = load_imf(
    new_net_path, 
    return_cfg=False
    #, ckpt_fpath = new_net_path + "/checkpoints/epoch_3_iters_4000.pt"
    )
mesh = imf2mesh(net, res = 64, normalize=True, bound = 1.3, threshold=0.0)

point = mesh.triangles_center
base = tens(point)
#u = net(base)

#i = np.argmax(torch.norm(gradient(u, base), dim = -1).detach().cpu())
#point = mesh.triangles_center[i]
#print(point)
base = torch.tensor(np.float32(point))
base.requires_grad = True
normal = mesh.face_normals
val_sum = 0
res = [None]*len(normal)
for j in range(len(normal)):
    val = [None]*100
    dist = [None]*100
    for i in range(100):
        val[i] = point[j] + 1/100*(i/50-1)*(normal[j])
        dist[i] = 1/100*(i/50-1)
    val = tens(val).view(100,3)

    vals = np.abs(net(val).detach().cpu().numpy() - dist).mean()
    val_sum += vals
    res[j] = [point[j][0], point[j][1], point[j][2], vals]
print(val_sum/len(normal))
np.savetxt("SDF_ptws_test.csv", res , delimiter = ",", header = "x,y,z,a")
