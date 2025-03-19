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
from trainers.utils.diff_ops import gradient, laplace, lapl_beltrami, divergence
import numpy as np
from trainers import RHS, analyticSDFs
from notebooks import error_utils
bs = 1000
new_net_path = "/home/weidemaier/PDE Net/logs/NeuralSDFs_2025-Mar-03-12-02-34"
net, cfg = load_imf(
    new_net_path, 
    return_cfg=False
    )

mesh = imf2mesh(net, res = 128, normalize=True, bound = 1.3, threshold=0.0)

verts = error_utils.loading_pts("/home/weidemaier/PDE Net/NFGP/sphere4.csv") #mesh.vertices
bs = verts.shape[0]
xyz = tens(verts)
normals = (xyz / torch.norm(xyz, dim = -1).view(bs,1)).view(bs,3) 
normals = (gradient(net(xyz), xyz)/torch.norm(gradient(net(xyz), xyz), dim = -1).view(bs,1)).view(bs,3)
print(normals[0])
print((gradient(net(xyz), xyz))[0])
curv = divergence(normals, xyz)
print(curv)
vec = torch.cat((xyz, curv), 1)
#curv = laplace(net(xyz), xyz)
curv = curv.detach().cpu().numpy()

np.savetxt("curv_test.csv", vec.detach().cpu() , delimiter = ",", header = "x,y,z,a")

normals = [None]*bs

i = 0
while i < bs: 
    normals[i] = [curv[i].item(),0,0 ]
    i+=1

write_obj(new_net_path + "/curvature_example.obj", mesh.vertices, mesh.faces, vertex_normals = normals)