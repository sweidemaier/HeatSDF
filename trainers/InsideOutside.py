import numpy as np
#import plotly.graph_objects as go
from scipy.ndimage import label

'''# Increase number of points to get a more continuous sphere surface
np.random.seed(42)
num_points = 5000
theta = np.random.uniform(0, 2 * np.pi, num_points)
phi = np.random.uniform(0, np.pi, num_points)
r = np.ones(num_points)  # Points on unit sphere
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
point_cloud = np.column_stack((x, y, z))
'''
def inside_outside(point_cloud, grid_size = 32):
    ###
    # Compute bounding box
    min_bounds = point_cloud.min(axis=0) - np.array([1.,1.,1.])
    max_bounds = point_cloud.max(axis=0) + np.array([1.,1.,1.])
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
    inside_real = inside_coords * grid_step + min_bounds + grid_step/2.
    occupied_real = occupied_coords * grid_step + min_bounds + grid_step/2.
    outside_real = outside_coords * grid_step + min_bounds + grid_step/2.
    np.savetxt("gt_inner.csv", inside_real, delimiter = ",", header = "x,y,z")
    np.savetxt("gt_outer.csv", outside_real , delimiter = ",", header = "x,y,z")
    return inside_real, outside_real