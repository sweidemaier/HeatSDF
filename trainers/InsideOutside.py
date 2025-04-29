import numpy as np
import torch

def inside_outside(point_cloud, grid_size = 32): #TODO remove if not used 
    ###
    # Compute bounding box
    min_bounds = point_cloud.min(axis=0) - np.array([1.,1.,1.])
    max_bounds = point_cloud.max(axis=0) + np.array([1.,1.,1.])
    min_bounds = np.array([-1.2, -1.2, -1.2])
    max_bounds = np.array([1.2, 1.2, 1.2])
    bbox_size = max_bounds - min_bounds

    # Define grid step
    grid_step = bbox_size / (grid_size - 1)

    def get_grid_index(point):
        idx = ((point - min_bounds) / grid_step).astype(int)
        # Clamp indices to valid range in case of rounding issues
        return tuple(np.clip(idx, 0, grid_size - 1))

    # Initialize grid: 0 means empty, 1 will mean occupied
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    # Mark cells containing points (occupied)
    for point in point_cloud:
        idx = get_grid_index(point)
        grid[idx] = 1

    # Flood fill to mark outside cells (only fill cells that are empty)
    outside = np.zeros_like(grid, dtype=bool)
    def flood_fill(x, y, z):
        stack = [(x, y, z)]
        while stack:
            cx, cy, cz = stack.pop()
            if (0 <= cx < grid_size and 0 <= cy < grid_size and 0 <= cz < grid_size and
                not outside[cx, cy, cz] and grid[cx, cy, cz] == 0):
                outside[cx, cy, cz] = True
                # Add 6-connected neighbors
                stack.extend([(cx+1, cy, cz), (cx-1, cy, cz),
                            (cx, cy+1, cz), (cx, cy-1, cz),
                            (cx, cy, cz+1), (cx, cy, cz-1)])

    #flood_fill(0, 0, 0)

    for x in range(grid_size):
        for y in range(grid_size):
            for z in [0, grid_size - 1]:
                if grid[x, y, z] == 0 and not outside[x, y, z]:
                    flood_fill(x, y, z)
    for x in range(grid_size):
        for z in range(grid_size):
            for y in [0, grid_size - 1]:
                if grid[x, y, z] == 0 and not outside[x, y, z]:
                    flood_fill(x, y, z)
    for y in range(grid_size):
        for z in range(grid_size):
            for x in [0, grid_size - 1]:
                if grid[x, y, z] == 0 and not outside[x, y, z]:
                    flood_fill(x, y, z)

    # Remaining cells (empty and not marked outside) are considered inside
    inside = (grid == 0) & (~outside)

    # Extract grid indices for each category
    inside_coords = np.array(np.where(inside)).T
    occupied_coords = np.array(np.where(grid == 1)).T
    outside_coords = np.array(np.where(outside)).T
    
    # Convert grid indices to real coordinates (voxel centers)
    print(grid_step/2)
    inside_real = inside_coords * grid_step + min_bounds + grid_step/2.
    occupied_real = occupied_coords * grid_step + min_bounds + grid_step/2. 
    outside_real = outside_coords * grid_step + min_bounds + grid_step/2.
    
    return inside_real, outside_real, occupied_real


def inside_outside_torch(point_cloud, grid_size=32, bounds=None):
    """
    Args:
        point_cloud: (N, 3) torch tensor (cuda) of points
        grid_size: size of voxel grid per axis
        bounds: (min_bound, max_bound) as tuples or tensors
    Returns:
        inside_real, outside_real, occupied_real: real-space voxel center coordinates
    """
    assert point_cloud.is_cuda, "Input point cloud must be on CUDA"

    device = point_cloud.device
    dtype = point_cloud.dtype

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

    # Flood fill from boundary
    outside = torch.zeros_like(grid)
    visited = torch.zeros_like(grid)

    # Get 6-neighbors
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

    return inside_real, outside_real, occupied_real
