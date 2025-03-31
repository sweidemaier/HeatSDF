import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import torch
import trimesh
from trainers.utils.new_utils import tens
from trainers.utils.vis_utils import imf2mesh
from notebooks import error_evals
from trainers import RHS, analyticSDFs
from utils import load_imf, write_obj
from trainers.utils.diff_ops import gradient
import numpy as np
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
new_net_path = "/home/weidemaier/PDE Net/logs/"
stri = ("NeuralSDFs2_2025-Mar-24-09-28-22", "NeuralSDFs2_2025-Feb-28-12-52-07", "NeuralSDFs2_2025-Feb-28-10-01-23", "NeuralSDFs2_2025-Feb-28-11-37-21")
#stri = ("NeuralSDFs2_2025-Mar-08-00-37-53", "NeuralSDFs2_2025-Mar-08-00-00-03", "NeuralSDFs2_2025-Mar-07-23-21-32", "NeuralSDFs2_2025-Mar-07-22-43-21", "NeuralSDFs2_2025-Mar-07-21-40-17", "NeuralSDFs2_2025-Mar-07-20-09-42", "NeuralSDFs2_2025-Mar-07-18-44-42", "NeuralSDFs2_2025-Mar-07-17-19-55", "NeuralSDFs2_2025-Mar-07-15-54-45", "NeuralSDFs2_2025-Mar-07-14-22-27")
stri = ("NeuralSDFs_2025-Mar-07-20-56-20", "NeuralSDFs_2025-Mar-07-19-28-07", "NeuralSDFs_2025-Mar-07-18-02-30", "NeuralSDFs_2025-Mar-07-16-37-03", "NeuralSDFs_2025-Mar-07-15-08-55", "NeuralSDFs_2025-Mar-07-14-35-33")
#stri = ("logs/NeuralSDFs2_2025-Feb-26-16-14-12", "logs/NeuralSDFs2_2025-Feb-26-18-16-03", "logs/NeuralSDFs2_2025-Feb-26-20-18-21", "logs/NeuralSDFs2_2025-Feb-26-22-34-31", "logs/NeuralSDFs2_2025-Feb-27-00-21-17", "logs/NeuralSDFs2_2025-Feb-27-01-26-44", "logs/NeuralSDFs2_2025-Feb-27-02-15-15")
#stri = ("NeuralSDFs2_2025-Feb-27-11-45-26", "NeuralSDFs2_2025-Feb-27-13-45-44", "NeuralSDFs2_2025-Feb-27-16-24-46" )
stri = ("NeuralSDFs_2025-Mar-10-23-12-43", "NeuralSDFs_2025-Mar-10-22-30-45", "NeuralSDFs_2025-Mar-10-21-48-36", "NeuralSDFs_2025-Mar-10-21-06-00", "NeuralSDFs_2025-Mar-10-20-23-50", "NeuralSDFs_2025-Mar-10-19-41-16", "NeuralSDFs_2025-Mar-10-18-58-57", "NeuralSDFs_2025-Mar-10-18-16-43", "NeuralSDFs_2025-Mar-10-17-34-04", "NeuralSDFs_2025-Mar-10-16-51-50")

stri = ("NeuralSDFs3_2025-Mar-22-16-52-39", "")#NeuralSDFs_2025-Mar-12-18-26-51/", "NeuralSDFs_2025-Mar-12-17-44-03/", "NeuralSDFs_2025-Mar-12-16-57-02/", "NeuralSDFs_2025-Mar-12-15-12-40/")
#stri = ("NeuralSDFs2_2025-Mar-16-17-55-01", "NeuralSDFs2_2025-Mar-16-19-04-23", "NeuralSDFs_2025-Mar-13-09-54-59", "NeuralSDFs_2025-Mar-13-12-18-19", "NeuralSDFs_2025-Mar-13-10-38-51", "NeuralSDFs_2025-Mar-12-19-09-49", "NeuralSDFs_2025-Mar-12-18-26-51", "NeuralSDFs_2025-Mar-12-17-44-03", "NeuralSDFs_2025-Mar-12-16-57-02", "NeuralSDFs_2025-Mar-12-15-12-40")
stri = ("NeuralSDFs3_2025-Mar-23-01-28-35", "NeuralSDFs3_2025-Mar-23-00-33-28", "NeuralSDFs3_2025-Mar-22-23-38-51", "NeuralSDFs3_2025-Mar-22-22-43-57", "NeuralSDFs3_2025-Mar-22-22-02-25", "NeuralSDFs3_2025-Mar-22-21-20-50", "NeuralSDFs3_2025-Mar-22-20-39-23", "NeuralSDFs3_2025-Mar-22-19-57-55", "NeuralSDFs3_2025-Mar-22-19-17-31", "NeuralSDFs3_2025-Mar-22-18-37-43", "NeuralSDFs3_2025-Mar-22-17-57-16", "NeuralSDFs3_2025-Mar-22-17-16-42")
stri = ("NeuralSDFs3_2025-Mar-23-14-56-55", "NeuralSDFs3_2025-Mar-23-14-36-39", "NeuralSDFs3_2025-Mar-23-14-16-24", "NeuralSDFs3_2025-Mar-23-13-56-06")
stri = ("NeuralSDFs_2025-Mar-25-07-59-58", "NeuralSDFs3_2025-Mar-25-07-55-03", "NeuralSDFs_2025-Mar-25-05-08-15", "NeuralSDFs3_2025-Mar-25-04-20-02", "NeuralSDFs_2025-Mar-25-02-23-41", "NeuralSDFs3_2025-Mar-25-01-11-23", "NeuralSDFs_2025-Mar-24-23-37-00", "NeuralSDFs3_2025-Mar-24-21-04-03", "NeuralSDFs_2025-Mar-24-20-21-17", "NeuralSDFs3_2025-Mar-24-17-20-05", "NeuralSDFs_2025-Mar-24-17-19-30")
###armadillo finals
stri = ("NeuralSDFs3_2025-Mar-26-00-14-17", "NeuralSDFs3_2025-Mar-25-23-36-35", "NeuralSDFs3_2025-Mar-25-22-58-58", "NeuralSDFs3_2025-Mar-25-22-21-21", "NeuralSDFs3_2025-Mar-25-21-43-40", "NeuralSDFs3_2025-Mar-25-21-05-52", "NeuralSDFs3_2025-Mar-25-20-28-16", "NeuralSDFs3_2025-Mar-25-19-50-34", "NeuralSDFs3_2025-Mar-25-19-12-53", "NeuralSDFs3_2025-Mar-25-18-35-13", "NeuralSDFs3_2025-Mar-25-17-57-23", "NeuralSDFs3_2025-Mar-25-17-19-33", "NeuralSDFs3_2025-Mar-25-16-33-30", "NeuralSDFs3_2025-Mar-25-15-43-49", "NeuralSDFs3_2025-Mar-25-14-53-40", "NeuralSDFs3_2025-Mar-25-13-24-56")
###heads finals
stri = ("NeuralSDFs_2025-Mar-27-01-01-23", "NeuralSDFs_2025-Mar-27-00-12-17", "NeuralSDFs_2025-Mar-26-23-23-14", "NeuralSDFs_2025-Mar-26-22-34-18", "NeuralSDFs_2025-Mar-26-21-45-20", "NeuralSDFs_2025-Mar-26-20-56-14", "NeuralSDFs_2025-Mar-26-20-07-11", "NeuralSDFs_2025-Mar-26-19-18-08", "NeuralSDFs_2025-Mar-26-18-29-01", "NeuralSDFs_2025-Mar-26-17-39-57", "NeuralSDFs_2025-Mar-26-16-50-52", "NeuralSDFs_2025-Mar-26-15-57-31")
stri = ("lightbulb","")
###armadillos 2
stri = ("NeuralSDFs3_2025-Mar-29-03-32-21", "NeuralSDFs3_2025-Mar-29-02-45-24", "NeuralSDFs3_2025-Mar-29-01-58-34", "NeuralSDFs3_2025-Mar-29-01-11-43", "NeuralSDFs3_2025-Mar-29-00-24-51", "NeuralSDFs3_2025-Mar-28-23-38-05", "NeuralSDFs3_2025-Mar-28-22-51-15", "NeuralSDFs3_2025-Mar-28-21-23-07", "NeuralSDFs3_2025-Mar-28-19-05-45", "NeuralSDFs3_2025-Mar-28-18-16-58", "NeuralSDFs3_2025-Mar-28-17-30-14", "NeuralSDFs3_2025-Mar-28-16-43-28", "NeuralSDFs3_2025-Mar-28-15-56-40", "NeuralSDFs3_2025-Mar-28-15-09-51", "NeuralSDFs3_2025-Mar-28-14-23-02")
###heads2
stri = ("NeuralSDFs3_2025-Mar-29-15-47-10", "NeuralSDFs3_2025-Mar-29-14-58-00", "NeuralSDFs3_2025-Mar-29-14-08-47", "NeuralSDFs3_2025-Mar-29-13-19-43", "NeuralSDFs3_2025-Mar-29-12-30-31", "NeuralSDFs3_2025-Mar-29-11-41-20", "NeuralSDFs3_2025-Mar-29-10-52-10", "NeuralSDFs3_2025-Mar-29-10-03-03", "NeuralSDFs3_2025-Mar-29-09-13-52", "NeuralSDFs3_2025-Mar-29-08-24-50", "NeuralSDFs3_2025-Mar-29-07-35-45", "NeuralSDFs3_2025-Mar-29-06-46-37", "NeuralSDFs3_2025-Mar-29-05-57-29", "NeuralSDFs3_2025-Mar-29-05-08-19", "NeuralSDFs3_2025-Mar-29-04-19-14")


for i in range(len(stri)):
    net, cfg = load_imf(
        new_net_path+stri[i], 
        return_cfg=False
        #,ckpt_fpath = new_net_path + stri[i] + "/checkpoints/epoch_10_iters_11000.pt"
    )
    bs = 10000
    mesh = imf2mesh(net, res = 512, normalize=True, bound = 1.15, threshold=0.0)
    trimesh.exchange.export.export_mesh(mesh, "/home/weidemaier/PDE Net/logs/" + str(stri[i]) + "/vis" + ".obj", file_type=None, resolver=None) #"+ str(i) + ".

    L2, chamfer, near, med, far, glob = error_evals.eval("logs/" + stri[i], mesh)

    with open(new_net_path  + "overview.txt", "a") as f:
        f.write(str(cfg.input.parameters.param1) +" / " + str(cfg.input.parameters.param2)+" & "+str(f"{L2:.4g}")+" / "+str(f"{chamfer:.4g}")  +" & "+str(f"{near:.4g}")+" / "+str(f"{med:.4g}") +" / "+ str(f"{far:.4g}")+" / "+ str(f"{glob:.4g}")+" & "+"\\\ "+" % "+ stri[i])

        f.write("\n")

    f = open(new_net_path + "overview.txt","r")
'''
count = 100
lspan = np.linspace(-1.3, 1.3, count)
sample_size = count**2
vec = [None]*sample_size
vec_2 = [None]*sample_size
i = 0
k = 0
while i < count:
    #x = r[i]*np.cos(phi[i])
    #y = r[i]*np.sin(phi[i])
    #z = 0
    #vec[i] = [x,y]
    #vec_2 = [np.cos(phi[i]), np.sin(phi[i])]
    j = 0
    while j < count:
        vec[i*count+j] = [lspan[i], lspan[j], 0]
        j = j+1
    
    i = i+1
#print(vec)
xyz_list = np.float32(vec)
xyz = torch.tensor(xyz_list)
xyz = xyz.cuda()
xyz.requires_grad = True



vec = net(xyz)
a = vec.detach().cpu().numpy()
#print(xyz.detach().cpu().numpy())
A = xyz.detach().cpu().numpy()
#print(A[:, 0], A[:,1], a[:,0])
i = 0
scale = max(a[:,0])
vec = [None]*sample_size
while i < sample_size:
    vec[i] = [A[i, 0], A[i,1], a[i,0]]
    i = i+1
#name = ("plot_2d" + ind)

np.savetxt("tests%d.csv", vec , delimiter = ",", header = "x,y,z")


#mesh = imf2mesh(analyticSDFs.comp_FEM, res = 256, normalize=True, bound = 2.3, threshold=0.0)

#xyz = tens(mesh.vertices)
#print(torch.abs(analyticSDFs.comp_FEM(xyz)).mean())
#print(torch.abs(net(xyz)).mean())
mesh = imf2mesh(net, res = 256, normalize=True, bound = 1.3, threshold=0.0)

phi = net
sample_size = 10000
box_width = 2.15
sample_l = tens(np.random.uniform(-box_width, box_width, (sample_size,1)))
sample_w = tens(np.random.uniform(-box_width, box_width, (sample_size,1)))
sample_h = tens(np.random.uniform(-box_width, box_width, (sample_size,1)))
sample = torch.cat((sample_l, sample_w, sample_h), 1)

#sample = tens(np.random.uniform(-box_width,box_width, (sample_size,3)))

phi_sort, indices = torch.sort(torch.abs(phi(sample)), dim = 0)              
t = torch.max((phi_sort < 0.1).nonzero(as_tuple=True)[0])
indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
xy = sample[indices[indices_to_keep]].reshape(t+1, 3)
true_bs = xy.shape[0]

while (true_bs < bs):
    sample = tens(np.random.uniform(-box_width,box_width, (sample_size,3)))
    phi_sort, indices = torch.sort(torch.abs(phi(sample)), dim = 0)
    #sample = sample[indices].squeeze()
    
    if((phi_sort < 0.1).nonzero(as_tuple=True)[0].shape[0] > 0 and (phi_sort < 0.1).nonzero(as_tuple=True)[0].shape[0] < sample_size):
        t = torch.max((phi_sort < 0.1).nonzero(as_tuple=True)[0])
        indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
        sample = sample[indices[indices_to_keep]].reshape(t+1, 3)
        
        xy= torch.cat((xy, sample), 0)
        true_bs = xy.shape[0]
bs = true_bs'''
'''
r = np.random.uniform(0,1.2, bs)
phi = np.random.uniform(0, 2*np.pi, bs)
theta = np.random.uniform(0, 2*np.pi, bs)
i = 0
random_points = [None]*bs
while (i < bs):
    x = r[i]*np.sin(theta[i])*np.cos(phi[i])
    y = r[i]*np.sin(theta[i])*np.sin(phi[i])
    z = r[i]*np.cos(theta[i])
    random_points[i] = [x,y,z]
    i = i+1
'''

#xyz_sample = tens(random_points)
#u = net(xy)
#grad_u_norm = torch.norm(gradient(u, xy), dim = -1)

###
###

#print(torch.square(grad_u_norm - 1).mean())
  