import os
import csv
import time
import torch
import numpy as np
import scipy.spatial as tree
import torch.nn.functional as F


### bump function with compact support
def bump_func(x):
    if (abs(x) > 1):
        return 0
    else:
        return np.exp(1/((abs(x)**2)-1))   



### compute locally adaptive weights for heat step
def comp_weights(pointcloud, epsilon, dim = 3):
    """
    Args:
        pointcloud: (N, D) numpy array of points in 2D or 3D
        epsilon: parameter determining the initial neighborhood sizefor points
        dim: spatial dimension (2 or 3)
    Returns:
        w: weights for locally adaptive reweighting during HeatStep
    """
    start = time.time()
    w = np.zeros(np.shape(pointcloud)[0])
    r = epsilon
    print("initial eps:", r)

    ### sort input points in tree
    tr = tree.cKDTree(pointcloud)
    p = tr.query_ball_point(x = pointcloud, r = r, workers = -1)
    
    ### increase radius, until for each point its eps-environment contains at least a few points; we choose 12
    while any(len(ball) < 12 for ball in p):
        r *= 2
        p = tr.query_ball_point(x = pointcloud, r = r, workers = -1)
    print("scaled eps:",r)
    
    c_eps = (r**dim)
    j = 0

    ### for each point compute weight
    while j < np.size(p):
        ball_indices = p[j]
        ball_points = pointcloud[ball_indices]
        dists = np.linalg.norm(pointcloud[j]-ball_points, axis = 1)
        sum = 1/c_eps*np.sum([bump_func(dists[i]/r) for i in range(len(dists))])
        w[j] = 1/sum
        j += 1
    
    ### normalize weights
    w = w/np.sum(w)
    print("Total computation time:", time.time() - start)
    return w



def sample_points_from_box_midpoints(box_midpoints, h, N, device='cuda', dim = 3):
    """
    Sample N points uniformly from non-overlapping 3D boxes

    Parameters:
        box_midpoints (torch.Tensor): Tensor of shape (B, 3), midpoints of each box.
        h (float): Edge length of the cube boxes.
        N (int): Number of points to sample.
        device (str): PyTorch device (e.g., 'cuda' or 'cpu').
        dim: Dimension

    Returns:
        torch.Tensor: Tensor of shape (N, 3) with sampled points.
    """
    B = box_midpoints.shape[0]

    # Sample box indices for each point
    box_indices = torch.randint(0, B, (N,), device=device)

    # Sample offsets in the range [-h/2, h/2]
    if dim == 2: 
        offsets = (torch.rand((N, 2), device=device) - 0.5) * h
    else: 
        offsets = (torch.rand((N, 3), device=device) - 0.5) * h

    # Gather midpoints for each selected box
    selected_midpoints = box_midpoints[box_indices]

    return selected_midpoints + offsets



def load_pts(cfg):
    ### load points from csv and scale to [-1,1]^3 
    with open(os.path.dirname(os.path.dirname(__file__)) + cfg.input.point_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            file = open(os.path.dirname(os.path.dirname(__file__)) + cfg.input.point_path)
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

    points -= np.mean(points, axis=0, keepdims=True)
    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.    
    points = np.float32(points)
    return points



def inside_outside_SDF(point_cloud, grid_size=32, bound=1.2, dilate=True, dim=3, safe_clouds = True):
    """
    Args:
        point_cloud: (N, D) torch tensor (cuda) of points in 2D or 3D
        grid_size: size of voxel grid per axis
        bounds: (min_bound, max_bound) as tuples or tensors
        dilate: whether to expand occupied voxels
        dim: spatial dimension (2 or 3)
    Returns:
        inside_real, outside_real, occupied_real: real-space voxel center coordinates
    """
    assert point_cloud.is_cuda, "Input point cloud must be on CUDA"
    assert dim in [2, 3], "Only 2D or 3D supported"

    device = point_cloud.device
    dtype = torch.float32

    # Default bounds
    
    min_bounds = torch.tensor([-bound]*dim, device=device, dtype=dtype)
    max_bounds = torch.tensor([bound]*dim, device=device, dtype=dtype)
    

    bbox_size = max_bounds - min_bounds
    grid_step = bbox_size / (grid_size - 1)

    # Compute voxel indices
    indices = torch.clamp(((point_cloud - min_bounds) / grid_step).long(), 0, grid_size - 1)

    if dim == 3:
        grid = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    else:  # 2D
        grid = torch.zeros((grid_size, grid_size), dtype=torch.bool, device=device)
        grid[indices[:, 0], indices[:, 1]] = True

    # === DILATION === #
    h = grid_step[0].item()
    dilate_count = max(1, int(np.ceil(0.1 / h)))

    grid_copy = grid.clone()
    if dilate:
        for i in range(dilate_count):
            if dim == 3:
                kernel = torch.ones((3, 3, 3), device=device)
                kernel[1, 1, 1] = 0
                kernel = kernel[None, None]
                grid_f = grid[None, None].float()
                padded = F.pad(grid_f, (1, 1, 1, 1, 1, 1))
                dilated = F.conv3d(padded, kernel) > 0
                grid_f = torch.logical_or(dilated, grid_f > 0).float()
                grid = grid_f[0, 0] > 0
            else:  # 2D
                kernel = torch.ones((3, 3), device=device)
                kernel[1, 1] = 0
                kernel = kernel[None, None]
                grid_f = grid[None, None].float()
                padded = F.pad(grid_f, (1, 1, 1, 1))
                dilated = F.conv2d(padded, kernel) > 0
                grid_f = torch.logical_or(dilated, grid_f > 0).float()
                grid = grid_f[0, 0] > 0

            if i == 0:
                grid_copy = grid.clone()

    # === FLOOD FILL === #
    visited = torch.zeros_like(grid)
    outside = torch.zeros_like(grid)

    if dim == 3:
        boundary_mask = torch.zeros_like(grid)
        boundary_mask[0, :, :] = boundary_mask[-1, :, :] = 1
        boundary_mask[:, 0, :] = boundary_mask[:, -1, :] = 1
        boundary_mask[:, :, 0] = boundary_mask[:, :, -1] = 1

        neighbors = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=device)
    else:
        boundary_mask = torch.zeros_like(grid)
        boundary_mask[0, :] = boundary_mask[-1, :] = 1
        boundary_mask[:, 0] = boundary_mask[:, -1] = 1

        neighbors = torch.tensor([
            [1, 0], [-1, 0],
            [0, 1], [0, -1]
        ], device=device)

    boundary_start = (~grid_copy) & boundary_mask
    queue = boundary_start.nonzero(as_tuple=False)
    visited[tuple(queue.T)] = True
    outside[tuple(queue.T)] = True

    # BFS
    while queue.numel() > 0:
        current = queue
        queue = []
        for offset in neighbors:
            neighbor_coords = current + offset
            mask = ((neighbor_coords >= 0) & (neighbor_coords < grid_size)).all(dim=1)
            neighbor_coords = neighbor_coords[mask]
            slices = tuple(neighbor_coords.T)
            new_mask = (~visited[slices]) & (~grid_copy[slices])
            visited[slices] = visited[slices] | new_mask
            outside[slices] = outside[slices] | new_mask
            queue.append(neighbor_coords[new_mask])
        if queue:
            queue = torch.cat(queue, dim=0)

    outside = outside & (~grid_copy)
    inside = (~grid_copy) & (~outside)

    def to_world(coords):
        return coords * grid_step + min_bounds + grid_step / 2.

    inside_real = to_world(inside.nonzero(as_tuple=False).float())
    outside_real = to_world(outside.nonzero(as_tuple=False).float())
    occupied_real = to_world(grid.nonzero(as_tuple=False).float())

    if safe_clouds:
        # Save
        np.savetxt("gt_inner.csv", inside_real.detach().cpu().numpy(), delimiter=",", header="x,y,z")
        np.savetxt("gt_outer.csv", outside_real.detach().cpu().numpy(), delimiter=",", header="x,y,z")
        np.savetxt("occupado.csv", occupied_real.detach().cpu().numpy(), delimiter=",", header="x,y,z")

    return inside_real, outside_real, occupied_real



