import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
from trainers.utils.diff_ops import manifold_gradient
from trainers.utils.new_utils import tens
def phi_plane(xyz):
    return xyz[:,2]

def phi_func_sphere(xyz):
    xyz = xyz.squeeze()
    norm = torch.norm(xyz, dim = -1, p = 2).view(xyz.shape[0], 1)
    return norm - torch.ones_like(norm) 

def phi_sphere_scaled(xyz):
    xyz = xyz.squeeze()
    norm = torch.norm(xyz, dim = -1, p = 2).view(xyz.shape[0], 1)
    return norm - 1.35*torch.ones_like(norm) 
def phi_func_sphere_band(xyz):
    norm = torch.norm(xyz, dim = -1, p = 2).view(xyz.shape[0], 1)
    return norm - torch.ones_like(norm) 
def phi_torus(xyz):
    r_1 = 0.5
    r_2 = 0.25
    xy = torch.cat((xyz[:,0].view(xyz.shape[0], 1), xyz[:,2].view(xyz.shape[0], 1)),1)
    q = torch.norm(xy, dim = -1).view(xyz.shape[0], 1) - r_1*torch.ones_like(torch.norm(xy, dim = -1).view(xyz.shape[0], 1))
    p = torch.cat((q.view(xyz.shape[0], 1), xyz[:,1].view(xyz.shape[0], 1)), 1)
    return torch.norm(p, dim = -1).view(xyz.shape[0], 1) - r_2*torch.ones_like(torch.norm(p, dim = -1).view(xyz.shape[0], 1))
def phi_link(xyz):
    l = 0.13
    r_1 = 0.2
    r_2 = 0.09
    q = torch.cat((xyz[:,0].reshape(xyz.shape[0], 1), torch.max(torch.abs(xyz[:,1].reshape(xyz.shape[0], 1) - l), torch.zeros((xyz.shape[0], 1)).cuda())), 1)
    vec_1 = torch.norm(q - r_1, dim = -1).view(xyz.shape[0], 1)
    vec_1 = torch.cat((vec_1, xyz[:,2].reshape(xyz.shape[0], 1)), 1)
    vec_2 = torch.norm(vec_1, dim = -1).view(xyz.shape[0], 1) - r_2
    return vec_2

def opTwist( primitive, p ):
    k = 10.0
    c = torch.cos(k*p[:,1])
    s = torch.sin(k*p[:,1])
    m1 = torch.cat((c.reshape(p.shape[0], 1),-s.reshape(p.shape[0], 1)), 1).reshape(p.shape[0], 2, 1)
    m2 = torch.cat((s.reshape(p.shape[0], 1),c.reshape(p.shape[0], 1)), 1).reshape(p.shape[0], 2, 1)
    m = torch.cat((m1, m2), 2)
    pxz = torch.cat((p[:,0].reshape(p.shape[0], 1), p[:,2].reshape(p.shape[0], 1)), 1)
    print(m)
    print(pxz.reshape(p.shape[0], 1, 2))
    print(torch.matmul(m,pxz.transpose(1,2)))
    q = torch.cat((m*pxz,p[:,1].reshape(p.shape[0], 1)), 2)
    return primitive(q)

def like_cheese(xyz):
    vec = torch.square(4*torch.square(xyz[:,0]) - torch.ones_like(xyz[:,0])) + torch.square(4*torch.square(xyz[:,1]) - torch.ones_like(xyz[:,0])) + torch.square(4*torch.square(xyz[:,2]) - torch.ones_like(xyz[:,0])) + 16*torch.square(torch.square(xyz[:,0]) + torch.square(xyz[:,1]) - torch.ones_like(xyz[:,0])) + 16*torch.square(torch.square(xyz[:,0]) + torch.square(xyz[:,2]) - torch.ones_like(xyz[:,0])) + 16*torch.square(torch.square(xyz[:,1]) + torch.square(xyz[:,2]) - torch.ones_like(xyz[:,0])) -16*torch.ones_like(xyz[:,0])
    return vec

def comp_FEM(xyz):
    xyz = xyz.squeeze()
    return 0.25*torch.square(xyz[:,0]) + torch.square(xyz[:,1]) + 4*torch.square(xyz[:,2])/torch.square((torch.ones_like(torch.sin(torch.pi*xyz[:,0])) + 0.5*torch.sin(torch.pi*xyz[:,0]))) -torch.ones(xyz.shape[0]).cuda()

def comp_FEM_scaled(xyz):
    xyz = xyz.squeeze()
    scale = torch.sqrt(5*(torch.ones_like(xyz[:,0])) + torch.square(0.5*((torch.ones_like(xyz[:,0])) + 0.5*torch.sin(2*torch.pi*xyz[:,0]))))
    return scale*(0.25*torch.square(xyz[:,0]) + torch.square(xyz[:,1]) + 4*torch.square(xyz[:,2])/torch.square((torch.ones_like(torch.sin(torch.pi*xyz[:,0])) + 0.5*torch.sin(torch.pi*xyz[:,0]))) -torch.ones(xyz.shape[0]).cuda())
def DF(xyz):
    i = 0
   
    DF = [None]*xyz.shape[0]
    while i < xyz.shape[0]:
        DF[i] = np.matrix([[2, 0, 0], [0, 1, 0], [0.5*xyz[i,2]*(np.pi*np.cos(2*np.pi*xyz[i,0])), 0, 0.5*(1 + 0.5*np.sin(2*np.pi*xyz[i,0]))]])
        #DF[i] = np.matrix([[2, 0, 0.5*xyz[i,2]*(np.pi*np.cos(2*np.pi*xyz[i,0]))], [0, 1, 0], [0 , 0, 0.5*(1 + 0.5*np.sin(2*np.pi*xyz[i,0]))]])
        i += 1
    return DF
def weightFEM(xyz, normals):
    xy = xyz.detach().cpu()
    df = DF(xy)
    df = tens(np.reshape(df, (xyz.shape[0], 3, 3)))
    #normals = xyz/torch.norm(xyz, dim = -1).view(xyz.shape[0], 1)
    DFn = torch.matmul(df, normals.reshape(xyz.shape[0], 3, 1)).reshape(xyz.shape[0], 3)
    DFn = torch.norm(DFn, dim = -1).view(xyz.shape[0], 1)
    #print(DFn)
    return (torch.abs(torch.ones_like(torch.sin(2*torch.pi*xyz[:,0])) + 0.5*torch.sin(2*torch.pi*xyz[:,0])).reshape(xyz.shape[0],1)/DFn.reshape(xyz.shape[0],1))
    

def weight_sphere(xyz):
   
    return torch.sin(xyz[:, 2])