import os
import csv
import igl
import time
import torch
import numpy as np
import scipy.spatial as tree
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader



### define smooth cut-off function
def eta(x, delta):
    vec = ((1/4)*(x/(delta) + 2*torch.ones(x.shape).cuda())*(x/(delta) - torch.ones(x.shape).cuda())**2)
    vec = torch.where(x <= -(delta)*torch.ones(x.shape).cuda(),  torch.ones(x.shape).cuda(), vec)
    vec = torch.where(x > (delta)*torch.ones(x.shape).cuda(), torch.zeros(x.shape).cuda(), vec)
    return vec.view(x.shape[0], 1)



### define smooth cut-off function
def beta(x, kappa):
    x = x/kappa
    vec = torch.where(x <= torch.zeros(x.shape).cuda(),  torch.zeros(x.shape).cuda(), -2*x**3 + 3*x**2)
    vec = torch.where(x > torch.ones(x.shape).cuda(), torch.ones(x.shape).cuda(), vec)
    return vec



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



def load_pts(cfg, extract_normals = False):
    ### Load points from CSV and scale to [-1,1]^3
    full_path = os.path.dirname(os.path.dirname(__file__)) + cfg.input.point_path

    with open(full_path, newline='') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        row_count = len(csv_reader)

        # Special case: exactly 6 rows, no header
        if extract_normals:
            count = row_count - 1
            points = [None] * count
            normals = [None]* count
            for i in range(1, row_count):
                a = float(csv_reader[i][0])
                na = float(csv_reader[i][3])
                b = float(csv_reader[i][1])
                nb = float(csv_reader[i][4])
                if cfg.models.decoder.dim == 3:
                    c = float(csv_reader[i][2])
                    nc = float(csv_reader[i][5])
                points[i - 1] = [a, b] if cfg.models.decoder.dim != 3 else [a, b, c]
                normals[i - 1] = [na, nb] if cfg.models.decoder.dim != 3 else [na, nb, nc]

            points = np.array(points)
            normals = np.array(normals)

            # Normalize points
            points -= np.mean(points, axis=0, keepdims=True)
            coord_max = np.amax(points)
            coord_min = np.amin(points)
            points = (points - coord_min) / (coord_max - coord_min)
            points -= 0.5
            points *= 2.
            points = np.float32(points)

            return points, np.float32(normals)

        # Default case: assume header + rows of points
        count = row_count - 1
        points = [None] * count
        for i in range(1, row_count):  # skip header
            a = float(csv_reader[i][0])
            b = float(csv_reader[i][1])
            if cfg.models.decoder.dim == 3:
                c = float(csv_reader[i][2])
            points[i - 1] = [a, b] if cfg.models.decoder.dim != 3 else [a, b, c]

        points = np.array(points)
        points -= np.mean(points, axis=0, keepdims=True)
        coord_max = np.amax(points)
        coord_min = np.amin(points)
        points = (points - coord_min) / (coord_max - coord_min)
        points -= 0.5
        points *= 2.
        points = np.float32(points)

        return points



def inside_outside_SDF(point_cloud, grid_size=32, bound=1.2, dilate=True, dim=3):
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

    return inside_real, outside_real, occupied_real


### for winding number computation
class BalancedInnerOuterDataset(Dataset):
    def __init__(self, inner_sample: torch.Tensor, outer_sample: torch.Tensor, num_batches: int = 1000):
        """
        inner_sample: [N_inner, 3]
        outer_sample: [N_outer, 3]
        num_batches: number of batches to cover entire outer_sample once
        """
        assert outer_sample.shape[0] >= num_batches, "Outer sample must be larger than number of batches"

        self.inner_sample = inner_sample
        self.outer_sample = outer_sample
        self.num_batches = num_batches

        self.batch_size = outer_sample.shape[0] // num_batches
        self.outer_indices = torch.randperm(outer_sample.shape[0])

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # Get outer batch deterministically
        start = idx * self.batch_size
        end = start + self.batch_size
        outer_idx = self.outer_indices[start:end]
        outer_batch = self.outer_sample[outer_idx]

        # Sample inner batch randomly of same size
        inner_idx = torch.randint(0, self.inner_sample.shape[0], (self.batch_size,))
        inner_batch = self.inner_sample[inner_idx]

        return inner_batch, outer_batch



### def routine to use surface normals to separate inner/outer region via winding-numbers
def comp_winding(net, tresh, N, P, domainbound = 1.2):
    k = 5  # or 20, depending on density
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(P)
    distances, _ = nbrs.kneighbors(P)

    # Estimate area as average squared distance to neighbors (skip self-distance at index 0)
    A_est = np.mean(distances[:, 1:]**2, axis=1)
    A = A_est[:, np.newaxis]  # shape (N, 1)
    # Step 1: Choose query points Q
    # For example: use a regular 3D grid bounding the point cloud´       
    grid_size = 120  # adjust as needed
    
    x = np.linspace(-domainbound, domainbound, grid_size)
    y = np.linspace(-domainbound, domainbound, grid_size)
    z = np.linspace(-domainbound, domainbound, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Q = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).astype(np.float32).T  # (N, 3)

    # Actual computation of winding numbers
    N = np.ascontiguousarray(N)
    A = np.ascontiguousarray(np.float32(A))
    Q = np.ascontiguousarray(Q)
    P = np.ascontiguousarray(P)

    w = igl.fast_winding_number_for_points(P, N, A, Q)
    w = w.reshape((grid_size, grid_size, grid_size))

    # Step 3: Threshold to determine inside/outside
    mask = (w > 0.5).reshape(-1)  
    inner_sample = torch.from_numpy(Q[mask]).to("cuda").float()
    mask = (net(inner_sample) < tresh).view(inner_sample.shape[0])
    inner_sample = inner_sample[mask, :]
    mask = (w < 0.5).reshape(-1)  
    outer_sample = torch.from_numpy(Q[mask]).to("cuda").float()
    mask = (net(outer_sample) < tresh).view(outer_sample.shape[0])
    outer_sample = outer_sample[mask, :]

    dataset = BalancedInnerOuterDataset(inner_sample, outer_sample, num_batches=1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return dataloader

