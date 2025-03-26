import scipy.spatial as tree
import scipy.integrate as integral
import numpy as np
import time
from trainers.utils.new_utils import tens
from trainers.utils.diff_ops import gradient
import torch

def eta(x, delta= 0.1):
    vec = (1/4)*(x/(delta) + 2*torch.torch.ones_like(x))*(x/(delta) - torch.ones_like(x))**2
    vec = torch.where(x <= -(delta)*torch.ones_like(x),  torch.ones_like(x), vec)
    vec = torch.where(x > (delta)*torch.ones_like(x), torch.zeros_like(x), vec)
    return vec.view(x.shape[0], 1)

def beta(x, kappa):
    x = x/kappa
    vec = torch.where(x <= torch.zeros_like(x),  torch.zeros_like(x), -2*x**3 + 3*x**2)
    vec = torch.where(x > torch.ones_like(x), torch.ones_like(x), vec)
    return vec

def bump_func(x):
        if (abs(x) > 1):
            return 0
        else:
            return np.exp(1/((abs(x)**2)-1))   
    
def comp_weights(pointcloud, epsilon, dim = 2 ):
    start = time.time()
    w = np.zeros(np.shape(pointcloud)[0])

    #c_eps = (epsilon**dim)
    tr = tree.cKDTree(pointcloud)
    
    r = epsilon
    print("initial eps:", r)
    p = tr.query_ball_point(x = pointcloud, r = r, workers = -1)
    while any(len(ball) < 12 for ball in p):
        r *= 2

        p = tr.query_ball_point(x = pointcloud, r = r, workers = -1)
    c_eps = (r**dim)
    j = 0
    print("scaled eps:",r)
    while j < np.size(p):
        ball_indices = p[j]
        ball_points = pointcloud[ball_indices]
        dists = np.linalg.norm(pointcloud[j]-ball_points, axis = 1)
        sum = 1/c_eps*np.sum([bump_func(dists[i]/r) for i in range(len(dists))])
        w[j] = 1/sum

        j += 1
    
    w = w/np.sum(w)
    print("Total computation time:", time.time() - start)
    return w

def comp_heat_gradients(gt_inner, gt_outer, near_net, far_net, kappa):
    #gt_outer = tens(gt_outer)
    #gt_inner = tens(gt_inner)
    outer_size = gt_outer.shape[0]
    inner_size = gt_inner.shape[0]
   
    grad_near_inner = gradient(near_net(gt_inner), gt_inner)
    grad_near_outer = gradient(near_net(gt_outer), gt_outer)
    grad_far_inner = gradient(far_net(gt_inner), gt_inner)
    grad_far_outer = gradient(far_net(gt_outer), gt_outer)
    
    u_near_inner = near_net(gt_inner)
    u_near_outer = near_net(gt_outer)

    n_inner = (1-beta(u_near_inner, kappa))*grad_far_inner + (beta(u_near_inner, kappa))*grad_near_inner
    n_inner = n_inner/torch.norm(n_inner, dim = -1).view(inner_size, 1)
    n_outer = (1-beta(u_near_outer, kappa))*grad_far_outer + (beta(u_near_outer, kappa))*grad_near_outer
    n_outer = n_outer/torch.norm(n_outer, dim = -1).view(outer_size, 1)
    n_outer = n_outer.detach()
    n_inner = n_inner.detach()
    
    return n_inner, n_outer

'''gt_outer = tens(gt_outer)
    gt_inner = tens(gt_inner)
    outer_size = gt_outer.shape[0]
    inner_size = gt_inner.shape[0]
   
    grad_near_inner = gradient(near_net(gt_inner), gt_inner)
    grad_near_outer = gradient(near_net(gt_outer), gt_outer)
    grad_far_inner = gradient(far_net(gt_inner), gt_inner)
    grad_far_outer = gradient(far_net(gt_outer), gt_outer)
    
    n_inner = beta(torch.norm(grad_near_inner, dim = -1), kappa = 500).view(inner_size, 1)*grad_far_inner + (1-beta(torch.norm(grad_near_inner, dim = -1), kappa = 500).view(inner_size, 1))*grad_near_inner
    n_inner = n_inner/torch.norm(n_inner, dim = -1).view(inner_size, 1)

    n_outer = beta(torch.norm(grad_near_outer, dim = -1), kappa = 500).view(outer_size, 1)*grad_far_outer + (1-beta(torch.norm(grad_near_outer, dim = -1), kappa = 500).view(outer_size, 1))*grad_near_outer
    n_outer = n_outer/torch.norm(n_outer, dim = -1).view(outer_size, 1)
    n_outer = n_outer.detach().cpu().numpy()
    n_inner = n_inner.detach().cpu().numpy()
    return n_inner, n_outer'''