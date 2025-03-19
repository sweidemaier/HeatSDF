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
mesh = imf2mesh(net, res = 256, normalize=True, bound = 1.3, threshold=0.0)

point = mesh.triangles_center
base = tens(point)
u = net(base)

i = np.argmax(torch.norm(gradient(u, base), dim = -1).detach().cpu())
point = mesh.triangles_center[i]
print(point)
base = torch.tensor(np.float32(point))
base.requires_grad = True
normal = mesh.face_normals[i]
val = [None]*100
dist = [None]*100
for i in range(100):
    val[i] = point + 1/100*(i/50-1)*(normal)
    dist[i] = 1/100*(i/50-1)

val = tens(val)
vals = net(val).detach().cpu()
grads = torch.norm(gradient(net(val), val), dim = -1).view(100,1).detach().cpu()
print(vals)
#plt.style.use('_mpl-gallery')
print(dist)
# make data
np.random.seed(1)
x = dist

# plot:
fig, ax = plt.subplots()

#ax.hist(x, bins=100, linewidth=0.5, edgecolor="white")
#ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       #ylim=(0, np.max(grads)), yticks=np.linspace(0,np.max(grads), 10))
#fig.savefig("tests.png", transparent=None)

ax.plot(x, vals, linewidth=2.0)
ax.plot(x, x, linewidth=2.0)

#ax.set(xlim=(-50/10000, 50/10000),
#       ylim=(0.98, 1.02))

fig.savefig("tests.png", transparent=None)
plt.show()


