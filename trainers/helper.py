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

    # Compute bounding box
    if bounds is None:
        min_bounds = torch.tensor([-1.2, -1.2, -1.2], device=device, dtype=dtype)
        max_bounds = torch.tensor([1.2, 1.2, 1.2], device=device, dtype=dtype)
    else:
        min_bounds, max_bounds = bounds

    bbox_size = max_bounds - min_bounds
    grid_step = bbox_size / (grid_size - 1)

    def get_grid_index(points):
        return torch.clamp(((points - min_bounds) / grid_step).long(), 0, grid_size - 1)

    # Create occupancy grid
    grid = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
    indices = get_grid_index(point_cloud)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # === DILATE OCCUPIED VOXELS === #
    if dilate:
        dilated_grid = grid.clone()
        neighbors = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=device)

        occ_coords = grid.nonzero(as_tuple=False)
        for offset in neighbors:
            neighbor_coords = occ_coords + offset
            # Clamp to valid indices
            valid_mask = ((neighbor_coords >= 0) & (neighbor_coords < grid_size)).all(dim=1)
            neighbor_coords = neighbor_coords[valid_mask]
            dilated_grid[neighbor_coords[:, 0], neighbor_coords[:, 1], neighbor_coords[:, 2]] = True

        grid = dilated_grid

    # Flood fill from boundary
    outside = torch.zeros_like(grid)
    visited = torch.zeros_like(grid)

    # 6-connected neighbors
    neighbors = torch.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], device=device)

    # Initialize queue with all boundary cells
    queue = []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in [0, grid_size-1]:
                for x, y, z in [(i, j, k), (i, k, j), (k, i, j)]:
                    if not grid[x, y, z] and not visited[x, y, z]:
                        visited[x, y, z] = True
                        outside[x, y, z] = True
                        queue.append((x, y, z))

    while queue:
        x, y, z = queue.pop()
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx.item(), y + dy.item(), z + dz.item()
            if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                if not visited[nx, ny, nz] and not grid[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    outside[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

    # Inside = not occupied and not outside
    inside = (~grid) & (~outside)

    def to_world(coords):
        return coords * grid_step + min_bounds + grid_step / 2.

    inside_real = to_world(inside.nonzero(as_tuple=False).float())
    outside_real = to_world(outside.nonzero(as_tuple=False).float())
    occupied_real = to_world(grid.nonzero(as_tuple=False).float())
    np.savetxt("gt_inner.csv", inside_real.detach().cpu().numpy() , delimiter = ",", header = "x,y,z")
    np.savetxt("gt_outer.csv", outside_real.detach().cpu().numpy() , delimiter = ",", header = "x,y,z")
    np.savetxt("occupado.csv", occupied_real.detach().cpu().numpy() , delimiter = ",", header = "x,y,z")
    return inside_real, outside_real, occupied_real

