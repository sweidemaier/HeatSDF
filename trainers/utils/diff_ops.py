# Based on https://github.com/vsitzmann/siren/blob/master/diff_operators.py
import torch
from torch.autograd import grad



### all diff-ops are based on torch autograd
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
    """
    Laplacian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, channels)
    """
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div



def divergence(y, x):
    """
    Divergence of y wrt x
    y: shape (meta_batch_size, num_observations, dim)  # vector field
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations)
    """
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div



def gradient(y, x, grad_outputs=None):
    """
    Gradient of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, dim, channels)
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
