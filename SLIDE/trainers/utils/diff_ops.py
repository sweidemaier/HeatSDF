# Based on https://github.com/vsitzmann/siren/blob/master/diff_operators.py
import torch
from torch.autograd import grad
import numpy as np

def hessian(y, x):
    """
    Hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, dim, channels)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations,
                    y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y,
                                   create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def laplace(y, x, normalize=False, eps=0., return_grad=False):
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div



def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div
def manifold_divergence(y, x, normal, grad_outputs=None):
    div = 0.
    div_n = 0.
    for i in range(y.shape[-1]):
        div += manifold_gradient(
            y[..., i], x, normal,)[..., i:i+1]
    #print(normal*div_n)
    #div -= torch.sum(normal*div,dim=0)    
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def manifold_gradient(y, x, normal, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    
    sum = torch.sum(normal * grad, dim=-1).reshape(y.shape[0],1)
    
    grad = grad - normal*sum
    return grad

def lapl_beltrami(y,x, normal, grad_outputs=None):
    grad = manifold_gradient(y, x, normal)
    lapl_b = manifold_divergence(grad, x, normal)
    return lapl_b
def manifold_jacobian(y, x, normal, grad_outputs = None):
    """
    Jacobian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    ret: shape (meta_batch_size, num_observations, channels, dim)
    """
    meta_batch_size, num_observations = y.shape[:2]
    
    # (meta_batch_size*num_points, 2, 2)
    jac = torch.zeros(
        meta_batch_size, num_observations,
        x.shape[-1]).to(y.device)

    
    if(y.shape!=(x.shape[0],3)):
        print("Error. Dims for Jacobian not implemented!")
        return
    

    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i] = grad(
            y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    sum1 = (torch.sum(normal * jac[:,0], dim=-1).reshape(y.shape[0],1)*normal).unsqueeze(1)
    sum2 = (torch.sum(normal * jac[:,1], dim=-1).reshape(y.shape[0],1)*normal).unsqueeze(1)
    sum3 = (torch.sum(normal * jac[:,2], dim=-1).reshape(y.shape[0],1)*normal).unsqueeze(1)

    sum = torch.cat((sum1, sum2, sum3),1)

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return (jac - torch.transpose(sum, 1,2))
def jacobian(y, x):
    """
    Jacobian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    ret: shape (meta_batch_size, num_observations, channels, dim)
    """
    meta_batch_size, num_observations = y.shape[:2]
    
    # (meta_batch_size*num_points, 2, 2)
    jac = torch.zeros(
        meta_batch_size, num_observations,
        x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i] = grad(
            y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac

def multi_jac(x,y,bs):
    outp = torch.autograd.functional.jacobian(x,y)
    
    #outp = jacobian(x[0], y[0])
    return outp