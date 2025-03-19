import os
import sys
import numpy as np
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import torch
from trainers.utils.new_utils import tens
from trainers import RHS, analyticSDFs
#from torchsummary import summary
import evaluation_surfaces
import trimesh
#from tvtk.api import tvtk, write_data
import pyevtk
from pyevtk.vtk import VtkTriangle, VtkQuad

new_net_path = "/home/weidemaier/PDE Net/NFGP/logs/create_neural_fields_2025-Mar-13-11-48-46"
mesh = "/home/weidemaier/PDE Net/logs/NeuralSDFs_2025-Mar-12-19-09-49/"
from trainers.utils.vis_utils import imf2mesh
from utils import load_imf_PDE, write_obj, load_imf
torch.set_grad_enabled(True)
net, cfg = load_imf_PDE(
    new_net_path, 
    return_cfg=False
    #, ckpt_fpath = new_net_path + "/best.pt"
    )
bs = 10000
mesh_net, cfg = load_imf(
    mesh, 
    return_cfg=False
    #, ckpt_fpath = new_net_path + "/best.pt"
    )
#xyz,_ = evaluation_surfaces.fibonacci_sphere(bs) #fibonacci_sphere(bs) #compFEM(bs)
#xyz = np.reshape(xyz, (bs, 3))
mesh = imf2mesh(mesh_net, res = 256, normalize=True, bound = 1.5, threshold=0.0)
print(mesh)
xyz = mesh.triangles_center
bs = xyz.shape[0]
print(bs)
#mesh.export('comp_FEM_mesh.stl')
#trimesh.exchange.export.export_mesh(mesh, "comparisonFEM_mesh.stl", file_type=None, resolver=None)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
i = 0
true_sol = [None]*bs

def RELU(x):
    if(np.max(x,0) == 0): print("0")
    return np.max(x,0)
while i < bs:
    r = 3
    #true_sol[i] = (RELU(0.5 + xyz[i,0]))**r + (RELU(0.5 - xyz[i,0]))**r + (RELU(0.5 + xyz[i,1]))**r + (RELU(0.5 - xyz[i,1]))**r + (RELU(0.5 + xyz[i,2]))**r + (RELU(0.5 - xyz[i,2]))**r
    true_sol[i] = xyz[i][0]*xyz[i][1]*xyz[i][2]
    i += 1


xyz = tens(xyz).reshape(bs, 3)

true_sol = tens(true_sol).reshape(bs, 1)

scale = net(xyz)[0]-true_sol[0] #torch.mean(net(xyz)-true_sol) #net(xyz)[0]-true_sol[0]# #torch.mean(net(xyz)-true_sol) #net(xyz)[0]-true_sol[0] #net(xyz).mean() #

#err_rel = torch.sqrt((torch.square(net(xyz).reshape(bs,1)-true_sol.reshape(bs,1) - scale*torch.ones(bs).cuda())).sum())/torch.sqrt((torch.square(true_sol + scale*torch.ones(bs).cuda())).sum())
err = torch.abs(net(xyz).reshape(bs,1)-true_sol.reshape(bs,1) - scale*torch.ones(bs,1).cuda())
print("L1-error:",err.mean().item())
#print("Relative L2-error:", err_rel)
print("L2:", torch.sqrt(torch.square(err).mean()).item())
print("L_inf:", torch.max(err).item())


#-net(xyz).mean()*torch.ones(bs).cuda().reshape(bs,1)
B = (net(xyz)).detach().cpu().numpy()
#C = true_sol.detach().cpu().numpy()

normals = [None]*bs

i = 0
while i < bs: 
    normals[i] = [B[i].item(), 0, 0 ]
    i+=1
write_obj(new_net_path + "/learned_1D_PDE.obj", mesh.vertices, mesh.faces, vertex_normals = normals)
faces = mesh.faces.reshape(3*mesh.faces.shape[0])
print("Conn",faces)
ctype = np.zeros(mesh.faces.shape[0])
offs = np.zeros(mesh.faces.shape[0])
for i in range (mesh.faces.shape[0]):
    ctype[i] = VtkTriangle.tid
    offs[i] = 3*(i+1)
#,cellData={'F': B}
print("ctype", ctype)
#offs = offs.astype(np.int64)

print(mesh.vertices[:,1])


print(mesh.faces.shape[0])
print(B.shape)
cellData = {"ec" : B.reshape(mesh.faces.shape[0])}
pyevtk.hl.unstructuredGridToVTK(new_net_path , np.ascontiguousarray(mesh.vertices[:,0]), np.ascontiguousarray(mesh.vertices[:,1]),np.ascontiguousarray(mesh.vertices[:,2]), connectivity=faces, offsets= offs, cell_types=ctype, cellData = cellData)
#pyevtk.hl.VtkStructuredGrid(new_net_path + "/learned_1D_PDE.vtk", mesh.vertices, mesh.faces, {'F': B})
#np.savetxt("True_sol_1D_PDE.csv",  vec, delimiter = ",", header = "x,y,z,e")

