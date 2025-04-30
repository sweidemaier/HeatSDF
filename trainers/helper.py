import scipy.spatial as tree
import scipy.integrate as integral
import numpy as np
import time
from trainers.utils.new_utils import tens
from trainers.utils.diff_ops import gradient
import torch
import csv
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
    outer_size = gt_outer.shape[0]
    inner_size = gt_inner.shape[0]
   
    grad_near_inner = gradient(near_net(gt_inner), gt_inner)
    grad_near_outer = gradient(near_net(gt_outer), gt_outer)
    if(far_net != None):
        grad_far_inner = gradient(far_net(gt_inner), gt_inner)
        grad_far_outer = gradient(far_net(gt_outer), gt_outer)
    
    u_near_inner = near_net(gt_inner)
    u_near_outer = near_net(gt_outer)
    if(far_net != None):
        n_inner = (1-beta(u_near_inner, kappa))*grad_far_inner + (beta(u_near_inner, kappa))*grad_near_inner
        n_outer = (1-beta(u_near_outer, kappa))*grad_far_outer + (beta(u_near_outer, kappa))*grad_near_outer
    else: 
        n_inner = grad_near_inner
        n_outer = grad_near_outer
    n_inner = n_inner/torch.norm(n_inner, dim = -1).view(inner_size, 1)
    n_outer = n_outer/torch.norm(n_outer, dim = -1).view(outer_size, 1)
    n_outer = n_outer.detach()
    n_inner = n_inner.detach()
    
    return n_inner, n_outer


def sample_points_from_box_midpoints(box_midpoints, h, N, device='cuda'):
    """
    Sample N points uniformly from non-overlapping 3D boxes using PyTorch (CUDA-compatible).

    Parameters:
        box_midpoints (torch.Tensor): Tensor of shape (B, 3), midpoints of each box.
        h (float): Edge length of the cube boxes.
        N (int): Number of points to sample.
        device (str): PyTorch device (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Tensor of shape (N, 3) with sampled points.
    """
    B = box_midpoints.shape[0]

    # Sample box indices for each point
    box_indices = torch.randint(0, B, (N,), device=device)

    # Sample offsets in the range [-h/2, h/2]
    offsets = (torch.rand((N, 3), device=device) - 0.5) * h

    # Gather midpoints for each selected box
    selected_midpoints = box_midpoints[box_indices]

    return selected_midpoints + offsets


def inside_outside_torch(point_cloud, grid_size=32, bounds=None, dilate=False):
    """
    Args:
        point_cloud: (N, 3) torch tensor (cuda) of points
        grid_size: size of voxel grid per axis
        bounds: (min_bound, max_bound) as tuples or tensors
        dilate: whether to expand occupied voxels to include neighbors
    Returns:
        inside_real, outside_real, occupied_real: real-space voxel center coordinates
    """
    assert point_cloud.is_cuda, "Input point cloud must be on CUDA"

    device = point_cloud.device
    dtype = torch.float32

    if bounds is None:
        min_bounds = torch.tensor([-1.2, -1.2, -1.2], device=device, dtype=dtype)
        max_bounds = torch.tensor([1.2, 1.2, 1.2], device=device, dtype=dtype)
    else:
        min_bounds, max_bounds = bounds

    bbox_size = max_bounds - min_bounds
    grid_step = bbox_size / (grid_size - 1)

    # Compute voxel indices
    indices = torch.clamp(((point_cloud - min_bounds) / grid_step).long(), 0, grid_size - 1)

    # Occupancy grid
    grid = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # === FLOOD FILL FOR OUTSIDE === #
    visited = torch.zeros_like(grid)
    outside = torch.zeros_like(grid)

    boundary_mask = torch.zeros_like(grid)
    boundary_mask[0, :, :] = boundary_mask[-1, :, :] = 1
    boundary_mask[:, 0, :] = boundary_mask[:, -1, :] = 1
    boundary_mask[:, :, 0] = boundary_mask[:, :, -1] = 1

    boundary_start = (~grid) & boundary_mask
    queue = boundary_start.nonzero(as_tuple=False)
    visited[queue[:, 0], queue[:, 1], queue[:, 2]] = True
    outside[queue[:, 0], queue[:, 1], queue[:, 2]] = True

    # BFS flood fill using queue
    neighbors = torch.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], device=device)

    while queue.numel() > 0:
        current = queue
        queue = []
        for offset in neighbors:
            neighbor_coords = current + offset
            mask = ((neighbor_coords >= 0) & (neighbor_coords < grid_size)).all(dim=1)
            neighbor_coords = neighbor_coords[mask]
            x, y, z = neighbor_coords[:, 0], neighbor_coords[:, 1], neighbor_coords[:, 2]
            new_mask = (~visited[x, y, z]) & (~grid[x, y, z])
            visited[x[new_mask], y[new_mask], z[new_mask]] = True
            outside[x[new_mask], y[new_mask], z[new_mask]] = True
            queue.append(neighbor_coords[new_mask])
        if queue:
            queue = torch.cat(queue, dim=0)

    # Inside = not occupied and not outside
    inside = (~grid) & (~outside)

    def to_world(coords):
        return coords * grid_step + min_bounds + grid_step / 2.

    inside_real = to_world(inside.nonzero(as_tuple=False).float())
    outside_real = to_world(outside.nonzero(as_tuple=False).float())

    # === DILATION (optional) === #
    if dilate:
        # Pad grid for convolution-style dilation
        kernel = torch.zeros((3, 3, 3), device=device)
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1
        kernel = kernel[None, None]

        grid_f = grid[None, None].float()  # Convert to float

        for _ in range(2):
            padded = F.pad(grid_f, (1, 1, 1, 1, 1, 1))
            dilated = F.conv3d(padded.float(), kernel) > 0
            grid_f = torch.logical_or(dilated, grid_f).float()
        grid = grid_f[0, 0] > 0

    occupied_real = to_world(grid.nonzero(as_tuple=False).float())
    return inside_real, outside_real, occupied_real


def load_pts(cfg): #TODO Florine hier gibts Probleme
    with open(cfg.input.point_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            file = open(cfg.input.point_path)
            count = len(file.readlines()) -1
            points = [None]*count
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    line_count += 1
                else:
                    a = float(row[0])
                    b = float(row[1])
                    if (cfg.models.decoder.dim == 3):
                        c = float(row[2])
                    points[line_count-1] = [a, b]
                    if (cfg.models.decoder.dim == 3):
                        points[line_count-1] = [a, b, c]
                    line_count += 1 
    if(cfg.input.normalize == "scale"):
        points -= np.mean(points, axis=0, keepdims=True)
        coord_max = np.amax(points)
        coord_min = np.amin(points)
        points = (points - coord_min) / (coord_max - coord_min)
        points -= 0.5
        points *= 2.    
    points = np.float32(points)
    return points