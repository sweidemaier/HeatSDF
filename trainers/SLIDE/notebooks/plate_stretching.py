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
new_net_path = "/home/weidemaier/PDE Net/NFGP/logs/create_neural_fields_2024-Nov-07-13-59-20"
import os.path as osp
from utils import load_imf
net, cfg = load_imf(
    new_net_path, 
    return_cfg=False)
force = cfg.models.decoder.force
E = 200.
h = 0.01
ny =0.3

D = 4*np.pi**4 #E*(2*h)**3/(12*(1-ny**2))

def sol(x,y):
    vec = np.sin(np.pi*x)*np.sin(np.pi*y)

    return vec
#evaluation for stretched plates
bs = 200
lsp = np.linspace(0,1,bs)
i = 0
j = 0
xy = [None]*(bs)**2
while i < bs:
    j = 0
    while j < bs:
        xy[i*bs + j] = [lsp[i], lsp[j]]
        j += 1
    i += 1

xyz = np.float32(xy)
xyz = torch.tensor(xyz)
xyz = xyz.cuda()
xyz.requires_grad = True
vec = net(xyz)

vec = vec.squeeze()
A = vec.detach().cpu().numpy()

i = 0

int = [None]*bs**2
KL = [None]*bs**2
truesol = [None]*bs**2
diff = [None]*bs**2
while i < bs:
    j = 0
    while j< bs: 
        #vec[i*100+j] = [xy[i*bs + j][0] + A[i*100 +j][0], xy[i*100 +j][1] + A[i*100 +j][1], 0]
        #int[i*100+j] = [xy[i*bs + j][0], xy[i*100 + j][1], np.linalg.norm(A, axis = 1)[i*100 + j]]
        #int[i*bs+j] = [xy[i*bs + j][0] + A[i*bs+j][0], xy[i*bs + j][1] + A[i*bs+j][1], A[i*bs+j][2]]
        KL[i*bs+j] = [xy[i*bs + j][0] , xy[i*bs + j][1] , A[i*bs+j]]
        truesol[i*bs+j] = [xy[i*bs + j][0] , xy[i*bs + j][1] , sol(xy[i*bs + j][0] , xy[i*bs + j][1])]
        diff [i*bs+j] = [xy[i*bs + j][0] , xy[i*bs + j][1] , sol(xy[i*bs + j][0] , xy[i*bs + j][1]) - A[i*bs + j]]
        j += 1
    i+=1
#np.savetxt("strechted.csv", vec , delimiter = ",", header = "x,y,z")
np.savetxt("deform.csv",  KL, delimiter = ",", header = "x,y,z")

np.savetxt("diff.csv",  diff, delimiter = ",", header = "x,y,z")
np.savetxt("deform_true.csv",  truesol, delimiter = ",", header = "x,y,z")
'''bs_bd = 10
bd_sample = np.random.uniform(0, 20/2, bs_bd)
zero_x = [None]*bs_bd
zero_y = [None]*bs_bd
i = 0
while i < bs_bd:
    zero_x[i] = [bd_sample[i], 0]
    zero_y[i] = [0, bd_sample[i]]
    i += 1

zero_x = tens(zero_x)
zero_y = tens(zero_y)
boundary_constr =  torch.abs(net(zero_x)[0][1]).mean() +  torch.abs(net(zero_y)[0][0]).mean()
print(boundary_constr.mean().item())'''
