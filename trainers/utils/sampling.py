import numpy as np
import os
import sys
import torch
import tqdm
from skimage import measure
from scipy.spatial import cKDTree
os.chdir("/home/weidemaier/PDE Net/NFGP") #TODO Florine pfad anpassen
print(os.getcwd())
sys.path.insert(0, os.getcwd())
from trainers.utils.vis_utils import imf2mesh
from trainers.analyticSDFs import comp_FEM #TODO Florine ??
from trainers.utils.new_utils import tens
from trainers.utils.diff_ops import gradient as autograd_gradient



# Step 2: Use Marching Cubes to extract the surface of the sphere #TODO Florine ??
def extract_surface(sdf, res = 30, bound= 1.1, level=0.0):
    """
    Extract the surface (isosurface) of the signed distance function using Marching Cubes.
    - sdf: The signed distance function.
    - level: The iso-level (we extract points at level 0, which corresponds to the 0-level set).
    Returns the vertices of the surface.
    Add random noise to a scalar field.
    - scalar_field: The input 3D array representing the scalar field.
    - noise_level: Magnitude of the random noise.
    Returns the noisy scalar field.
    """
    noise_level=0.1
    batch_size = 100
    
    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    grid = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float)
    grid = (grid / float(res - 1) - 0.5) * 2 * bound
    grid = grid.reshape(-1, 3)
    # Grid will be [-1, 1] * bound
    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    noise = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float)
    noise = (noise / float(res - 1) - 0.5) * 2 * bound
    noise = noise.reshape(-1, 3)
    grid = grid + noise_level*noise
    
    ###
    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    print(pbar)
    
    pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        
        xyz = torch.from_numpy(
            grid[sidx:eidx, :]).float().cuda().view(1, -1, 3)
        xyz.requires_grad = True
        distances = sdf(xyz)
        distances = distances.cpu().detach().numpy()
        dists_lst.append(distances.reshape(-1))

    dists = np.concatenate(
        [x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(-1)

    field = dists.reshape(res, res, res)
    vert, face, _, _ = measure.marching_cubes(
            field, level)
    vert = (vert / float(res - 1) - 0.5) * 2 * bound
    return vert
def lloyd_relaxation_with_repulsion(points, sdf_func, epsilon=0.1, iterations=10, k=10, alpha=0.1):
    """
    Perform Lloyd relaxation with a global repulsion force to ensure uniformity.
    
    Parameters:
    - points (ndarray): Nx3 array of 3D points.
    - sdf_func (function): Function to compute the SDF value at a given point.
    - epsilon (float): Band thickness around the 0-level set.
    - iterations (int): Number of relaxation iterations.
    - k (int): Number of nearest neighbors to consider.
    - alpha (float): Weight of the global repulsion force.
    
    Returns:
    - points (ndarray): Relaxed Nx3 array of 3D points.
    """
    points = np.copy(points)

    for _ in range(iterations):
        # Build KDTree for nearest-neighbor search
        tree = cKDTree(points)
        new_points = np.zeros_like(points)

        for i, point in enumerate(points):
            # Find k-nearest neighbors
            _, indices = tree.query(point, k=k)
            neighbors = points[indices]

            # Compute Lloyd centroid
            centroid = np.mean(neighbors, axis=0)

            # Compute global repulsion force
            repulsion_force = np.sum(
                (point - points) / (np.linalg.norm(point - points, axis=1, keepdims=True) ** 3 + 1e-6),
                axis=0,
            )

            # Update point position with both centroid and repulsion
            new_point = point + (centroid - point) + alpha * repulsion_force

            # Project back to surface
            sdf_value = sdf_func(new_point)
            gradient = new_point / np.linalg.norm(new_point)  # Approximate gradient
            new_point = new_point - sdf_value * gradient

            new_points[i] = new_point

        points = new_points

    return points
def lloyd_relaxation(sdf, points, epsilon=0.1, iterations=20):
    """
    Apply Lloyd's relaxation algorithm to distribute points uniformly on a surface.
    Points are projected back to the epsilon-thick band around the surface.
    - points: Initial points to relax.
    - center: Center of the surface (sphere).
    - epsilon: Thickness of the narrow band around the zero-level set.
    - iterations: Number of relaxation iterations.
    Returns the uniformly distributed points.
    """
    for _ in range(iterations):
        tree = cKDTree(points)
        new_points = []
        for point in points:
            # 1. Nächste Nachbarn finden
            _, idxs = tree.query(point, k=5)  # Verwende 6 Nachbarn

            neighbors = points[idxs]

            # 2. Schwerpunkt der Nachbarn berechnen
            centroid = np.mean(neighbors, axis=0)

            # 3. Punkt in Richtung des Schwerpunkts bewegen
            direction = centroid - point
            step = direction / np.linalg.norm(direction) * epsilon
            new_point = point + step

             # 4. Punkt auf das 0-Level-Set projizieren
            sdf_value = sdf(tens(point))
            gradient = autograd_gradient(sdf, point)#point / np.linalg.norm(point)
            projection = sdf_value * gradient
            projected_point = new_point - projection

            new_points.append(projected_point)

        points = np.array(new_points)
    return points

# Step 4: Uniformly sample points near the surface in an epsilon-thick band
def sample_points_around_surface(sdf, verts, epsilon=0.1):
    """
    Sample points in an epsilon-thick band around the surface of the sphere.
    - verts: Vertices extracted from Marching Cubes (surface points).
    - center: Center of the sphere.
    - epsilon: Thickness of the narrow band around the zero-level set.
    Returns a list of points sampled around the surface.
    """
    points = []
    for point in verts:
        # Compute the normal direction
        point = tens(point)
        grad = autograd_gradient(sdf(point), point)
        normal = grad/torch.norm(grad, dim = -1)
                
        # Sample points along the normal direction
        for delta in [epsilon, -epsilon]:  # Sample in both directions along the normal
            sampled_point = point + delta * normal
            points.append(sampled_point)
    return np.array(points)

# Step 5: Combine everything to get uniformly distributed points
def get_uniform_points(sdf, resolution, bound, epsilon=0.1, lloyd_iterations=2):
   
    # Step 1: Extract the surface using Marching Cubes
    surface_points = extract_surface(sdf, resolution, bound, level=0.0)
    
    # Step 2: Sample points around the surface in an epsilon-thick band
    sampled_points = sample_points_around_surface(sdf, surface_points, epsilon)

    # Step 3: Apply Lloyd's relaxation to distribute points uniformly
    uniform_points = lloyd_relaxation(sdf, sampled_points, epsilon, iterations=lloyd_iterations)

    # Return the uniformly distributed points
    return uniform_points

def sample(func, res, eps, bound):
    # Parameters
    epsilon = eps  # Narrow band thickness
    # Get the uniformly distributed points around the zero-level set
    uniform_points = get_uniform_points(sdf = func, resolution = res, epsilon=eps, bound = bound)
    np.savetxt("sampling.csv",  uniform_points, delimiter = ",", header = "x,y,z") #TODO Florine Pfad 
    return uniform_points

sample(comp_FEM, 30, 0.1, 2.1)