from trainers.utils.diff_ops import gradient
from trainers.utils.diff_ops import divergence as div
from trainers import analyticSDFs
import torch
from trainers.utils.new_utils import tens
def RHS_1(xyz, phi):
    return 12*xyz[:,0]*xyz[:,1]*xyz[:,2]
def RHS_1_proper(xyz, phi):
    xyz_norm_2 = torch.square(torch.norm(xyz, dim = -1, p = 2).view(xyz.shape[0], 1))
    return (12/xyz_norm_2)*xyz[:,0]*xyz[:,1]*xyz[:,2]

def RHS_2(xyz, phi = None):
    xyz = xyz.squeeze()
    phi = analyticSDFs.comp_FEM(xyz)
    grad_phi = gradient(phi, xyz).view(
                xyz.shape[0], 3)
    nu = grad_phi/torch.norm(grad_phi, dim = -1, p = 2).view(xyz.shape[0], 1)
    H = div(nu, xyz)
    return (2*nu[:,0].reshape(xyz.shape[0])*nu[:,1].reshape(xyz.shape[0]) + H.reshape(xyz.shape[0])*(xyz[:,1]*nu[:,0] + xyz[:,0]*nu[:,1])).reshape(xyz.shape[0], 1)

def RHS_2_imf(xyz):
    xyz = xyz.squeeze()
    phi = analyticSDFs.comp_FEM(xyz)
    grad_phi = gradient(phi, xyz).view(
                xyz.shape[0], 3)
    nu = grad_phi/torch.norm(grad_phi, dim = -1).view(xyz.shape[0], 1)
    
    H = div(nu, xyz)
    return (2*nu[:,0].reshape(xyz.shape[0])*nu[:,1].reshape(xyz.shape[0]) + H.reshape(xyz.shape[0])*(xyz[:,1]*nu[:,0] + xyz[:,0]*nu[:,1])).reshape(xyz.shape[0], 1)
def RELU(x):
    x = x.reshape(x.shape[0],1)
    return torch.where(x > torch.zeros_like(x),x, torch.zeros_like(x))
def RHS_nonsmooth(xyz, phi):
    r = 3
    sol_1 = ((RELU(0.5*torch.ones_like(xyz[:,0]) + xyz[:,0]))**r + (RELU(0.5*torch.ones_like(xyz[:,0]) - xyz[:,0]))**r)+ 2*r*xyz[:,0].reshape(xyz.shape[0],1)*((RELU(0.5*torch.ones_like(xyz[:,0]) + xyz[:,0]))**(r-1) - (RELU(0.5*torch.ones_like(xyz[:,0]) - xyz[:,0]))**(r-1)) -(r-1)*r*(1-(xyz[:,0].reshape(xyz.shape[0],1)**2).reshape(xyz.shape[0],1))*((RELU(0.5*torch.ones_like(xyz[:,0]) + xyz[:,0]))**(r-2) + (RELU(0.5*torch.ones_like(xyz[:,0]) - xyz[:,0]))**(r-2)).reshape(xyz.shape[0],1)
    sol_2 = ((RELU(0.5*torch.ones_like(xyz[:,1]) + xyz[:,1]))**r + (RELU(0.5*torch.ones_like(xyz[:,1]) - xyz[:,1]))**r)+ 2*r*xyz[:,1].reshape(xyz.shape[0],1)*((RELU(0.5*torch.ones_like(xyz[:,1]) + xyz[:,1]))**(r-1) - (RELU(0.5*torch.ones_like(xyz[:,1]) - xyz[:,1]))**(r-1)) -(r-1)*r*(1-(xyz[:,1].reshape(xyz.shape[0],1)**2).reshape(xyz.shape[0],1))*((RELU(0.5*torch.ones_like(xyz[:,1]) + xyz[:,1]))**(r-2) + (RELU(0.5*torch.ones_like(xyz[:,1]) - xyz[:,1]))**(r-2)).reshape(xyz.shape[0],1)
    sol_3 = ((RELU(0.5*torch.ones_like(xyz[:,2]) + xyz[:,2]))**r + (RELU(0.5*torch.ones_like(xyz[:,2]) - xyz[:,2]))**r)+ 2*r*xyz[:,2].reshape(xyz.shape[0],1)*((RELU(0.5*torch.ones_like(xyz[:,2]) + xyz[:,2]))**(r-1) - (RELU(0.5*torch.ones_like(xyz[:,2]) - xyz[:,2]))**(r-1)) -(r-1)*r*(1-(xyz[:,2].reshape(xyz.shape[0],1)**2).reshape(xyz.shape[0],1))*((RELU(0.5*torch.ones_like(xyz[:,2]) + xyz[:,2]))**(r-2) + (RELU(0.5*torch.ones_like(xyz[:,2]) - xyz[:,2]))**(r-2)).reshape(xyz.shape[0],1)

    return sol_1 + sol_2 + sol_3