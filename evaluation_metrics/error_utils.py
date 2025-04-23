import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import trimesh
from trainers.utils.new_utils import tens
import numpy as np
import csv
import mesh2sdf
import mesh_to_sdf
from utils import load_imf, write_obj
from trainers.utils.diff_ops import gradient

def loading_pts(filename, center = True):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        file = open(filename)
        count = len(file.readlines()) -1
        points = [None]*count
        line_count = 0
        for row in csv_reader:
            if (line_count == 0):
                line_count += 1
            else:
                a = float(row[0])
                b = float(row[1])
                c = float(row[2])
                points[line_count-1] = [a, b, c]
                line_count += 1 
    if (center):
        points -= np.mean(points, axis=0, keepdims=True)
        coord_max = np.amax(points)
        coord_min = np.amin(points)
        points = (points - coord_min) / (coord_max - coord_min)
        points -= 0.5
        points *= 2.    
    points = np.float32(points)

    return points



def band(phi,bs, width):
    sample_size = 10000
    box_width = 1.30
    sample_l = tens(np.random.uniform(-box_width, box_width, (sample_size,1)))
    sample_w = tens(np.random.uniform(-box_width, box_width, (sample_size,1)))
    sample_h = tens(np.random.uniform(-box_width, box_width, (sample_size,1)))
    sample = torch.cat((sample_l, sample_w, sample_h), 1)

    phi_sort, indices = torch.sort(torch.abs(phi(sample)), dim = 0)              
    t = torch.max((phi_sort < width).nonzero(as_tuple=True)[0])
    indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
    xy = sample[indices[indices_to_keep]].reshape(t+1, 3)
    true_bs = xy.shape[0]

    while (true_bs < bs):
        sample = tens(np.random.uniform(-box_width,box_width, (sample_size,3)))
        phi_sort, indices = torch.sort(torch.abs(phi(sample)), dim = 0)
        #sample = sample[indices].squeeze()
        
        if((phi_sort < width).nonzero(as_tuple=True)[0].shape[0] > 0 and (phi_sort < width).nonzero(as_tuple=True)[0].shape[0] < sample_size):
            t = torch.max((phi_sort < width).nonzero(as_tuple=True)[0])
            indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
            sample = sample[indices[indices_to_keep]].reshape(t+1, 3)
            
            xy= torch.cat((xy, sample), 0)
            true_bs = xy.shape[0]
    bs = true_bs
    return xy, bs

def get_gt(path, pts, center = False):
    mesh = trimesh.load(path, force='mesh')
    if (center == True): mesh = center_mesh(path)

    sdf = mesh_to_sdf.mesh_to_sdf(mesh, pts , surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
    return sdf

def center_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    points = mesh.vertices
    
    points -= np.mean(points, axis=0, keepdims=True)
    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.    
    points = np.float32(points)
    print(points)
    #trimesh.exchange.export.export_mesh(mesh, "/home/weidemaier/PDE Net/logs/" + "armadillo_center" + ".obj", file_type=None, resolver=None) #"+ str(i) + ".
    
    #write_obj("genus_dino_centered_mesh.obj", points, mesh.faces)
    return mesh
    
def get_normals(path):    
    mesh = trimesh.load(path, force='mesh')
    points = mesh.triangles_center
    normals = mesh.face_normals
    return normals, points
def gt_band(path,bs, width, sample_size = 5000):
    box_width = 1.2
    sample = np.random.uniform(-box_width, box_width, (sample_size,3))
    
    gt_values = get_gt(path, sample)
    sample = tens(sample)
    gt_values=tens(gt_values)
    phi_sort, indices = torch.sort(torch.abs(gt_values), dim = 0)              
    t = torch.max((phi_sort < width).nonzero(as_tuple=True)[0])
    indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
    xy = sample[indices[indices_to_keep]].reshape(t+1, 3)
    true_bs = xy.shape[0]
    print(true_bs)
    while (true_bs < bs):
        sample = np.random.uniform(-box_width,box_width, (sample_size,3))
        gt_values = get_gt(path, sample)
        gt_values=tens(gt_values)
        sample = tens(sample)
        phi_sort, indices = torch.sort(torch.abs(gt_values), dim = 0) 
         #sample = sample[indices].squeeze()
        
        if((phi_sort < width).nonzero(as_tuple=True)[0].shape[0] > 0 and (phi_sort < width).nonzero(as_tuple=True)[0].shape[0] < sample_size):
            t = torch.max((phi_sort < width).nonzero(as_tuple=True)[0])
            indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
            sample = sample[indices[indices_to_keep]].reshape(t+1, 3)
            
            xy= torch.cat((xy, sample), 0)
            true_bs = xy.shape[0]
            print(true_bs)
    bs = true_bs
    return xy, bs
def gt_bean(bs, width, bean, sample_size = 5000):
    box_width = 1.2
    sample = np.random.uniform(-box_width, box_width, (sample_size,3))
    
    gt_values = bean(sample[:,0], sample[:,1], sample[:,2])
    sample = tens(sample)
    gt_values=tens(gt_values)
    phi_sort, indices = torch.sort(torch.abs(gt_values), dim = 0)              
    t = torch.max((phi_sort < width).nonzero(as_tuple=True)[0])
    indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
    xy = sample[indices[indices_to_keep]].reshape(t+1, 3)
    true_bs = xy.shape[0]
    print(true_bs)
    while (true_bs < bs):
        sample = np.random.uniform(-box_width,box_width, (sample_size,3))
        gt_values = bean(sample[:,0], sample[:,1], sample[:,2])
        gt_values=tens(gt_values)
        sample = tens(sample)
        phi_sort, indices = torch.sort(torch.abs(gt_values), dim = 0) 
         #sample = sample[indices].squeeze()
        
        if((phi_sort < width).nonzero(as_tuple=True)[0].shape[0] > 0 and (phi_sort < width).nonzero(as_tuple=True)[0].shape[0] < sample_size):
            t = torch.max((phi_sort < width).nonzero(as_tuple=True)[0])
            indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
            sample = sample[indices[indices_to_keep]].reshape(t+1, 3)
            
            xy= torch.cat((xy, sample), 0)
            true_bs = xy.shape[0]
            print(true_bs)
    bs = true_bs
    return xy, bs
def chamfer_score(neural_mesh, gt_mesh, cloud_size = 5000):
    p = trimesh.sample.sample_surface_even(neural_mesh, cloud_size)
    q = trimesh.sample.sample_surface_even(gt_mesh, cloud_size)
    sum_1 = 0
    sum_2 = 0
    for j in range(cloud_size):    
        val = np.linalg.norm(p[j] - q, axis = 1, ord = 1).min()
        sum_1 += val
    for j in range(cloud_size):    
        val = np.linalg.norm(q[j] - p, axis = 1, ord = 1).min()
        sum_2 += val
    chamfer = 1/(2*cloud_size)*sum_1 + 1/(2*cloud_size)*sum_2
    return chamfer
def gt_inner_outer(path):
    sample_size = 10000
    box_width = 1.3
    sample = np.random.uniform(-box_width, box_width, (sample_size,3))
    
    gt_values = get_gt(path, sample, center=False)
    sample = tens(sample)
    gt_values=tens(gt_values)
    phi_sort, indices = torch.sort(gt_values, dim = 0)         
    print(phi_sort)     
    T = torch.min((phi_sort > 0.05).nonzero(as_tuple=True)[0])
    print(T)
    t = torch.max((phi_sort < -0.05).nonzero(as_tuple=True)[0])
    outer_indices = [i for i in range(phi_sort.shape[0]) if i >= T.item()]
    inner_indices = [j for j in range(phi_sort.shape[0]) if j <= t.item()]
    xy_inner = sample[indices[inner_indices]]#.reshape(t+1, 3)
    xy_outer = sample[indices[outer_indices]]#.reshape(T+1, 3)
    true_bs = xy_inner.shape[0]
    print(xy_inner.shape[0])
    print(xy_outer.shape[0])
    np.savetxt("gt_inner_head.csv", xy_inner.detach().cpu() , delimiter = ",", header = "x,y,z")
    np.savetxt("gt_outer_head.csv", xy_outer.detach().cpu() , delimiter = ",", header = "x,y,z")
    
    while (true_bs < 500):
        sample = np.random.uniform(-box_width,box_width, (sample_size,3))
        gt_values = get_gt(path, sample, center=False)
        sample = tens(sample)
        gt_values=tens(gt_values)
        phi_sort, indices = torch.sort(gt_values, dim = 0)         
        t = torch.max((phi_sort < -0.05).nonzero(as_tuple=True)[0])
        inner_indices = [j for j in range(phi_sort.shape[0]) if j <= t.item()]
        xy_inner = torch.cat((xy_inner,sample[indices[inner_indices]]), 0)#.reshape(t+1, 3)
        true_bs = true_bs + xy_inner.shape[0]
        print(true_bs)
    print(xy_inner.shape[0])
    print(xy_outer.shape[0])
    np.savetxt("gt_inner_head.csv", xy_inner.detach().cpu() , delimiter = ",", header = "x,y,z")
    np.savetxt("gt_outer_head.csv", xy_outer.detach().cpu() , delimiter = ",", header = "x,y,z")
        
    return xy_inner.detach().cpu(), xy_outer.detach().cpu()
    
def Eikonal_hist(current_path, object_type, net, far_band):

    eikonal = torch.norm(gradient(net(far_band), far_band), dim = -1)
    data = eikonal.detach().cpu().numpy() 

    bins = np.arange(0.01, 2.01, 0.01)  
    fig, ax = plt.subplots()
    # Create the histogram
    ax.hist(data, bins=bins, alpha=0.75,weights=np.ones_like(data) * 1/ len(data))


    # Labels and title
    plt.xlabel("Gradient Norm")
    plt.ylabel("Frequency")
    plt.title("Histogram of Eikonal Error - Ours: " + str(object_type))

    # Show grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    fig.savefig(current_path + "/eikonal_hist.png", transparent=None)
    plt.rcParams['svg.fonttype'] = 'none' # Use text as text in SVG
    plt.savefig(current_path + "/eikonal_hist" +'.svg', format = 'svg')   

def SDF_scatter(current_path, mesh_path, net, band_far, bean = False):
    bs = band_far.shape[0]
    ### difference to groundtruth SDF
    neural_dist = net(band_far)
    if(bean):
        np_band = band_far.detach().cpu().numpy()
        gt = sdf_capped_torus(np_band[:,0],np_band[:,1],np_band[:,2])
    else:
        gt = get_gt(mesh_path, band_far.detach().cpu().numpy())
    
    gt_res = np.reshape(gt, (bs,1))

    
    fig, ax = plt.subplots()
    ax.scatter(gt_res,neural_dist.detach().cpu().numpy() , s=0.1)
    min_val = gt.min()
    max_val = gt.max()
    # Plot the diagonal line
    plt.grid(visible=True)
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    plt.xlabel("ground truth distance")
    plt.ylabel("neural distance")
    ax.set_xlim(left=-0.1, right =0.1)
    ax.set_ylim(bottom=-0.15, top = 0.15)
    fig.savefig(current_path +"/_SDF_Linear_scatter.png", transparent=None)
    plt.rcParams['svg.fonttype'] = 'none' # Use text as text in SVG
    plt.savefig(current_path + "/_SDF_Linear_scatter" +'.svg', format = 'svg')  
    return gt_res, neural_dist.detach().cpu().numpy()

def L2_loss(mesh, net):
    on_surf_np = trimesh.sample.sample_surface_even(mesh, 50000)[0]
    on_surf = torch.tensor(np.float32(on_surf_np)).cuda()
    err_L2 = torch.square(net(on_surf)).mean()
    return err_L2
def chamfer_new_score(neural_mesh, gt_mesh, cloud_size=10000):
    # Randomly sample points from both sets
    p, _ = trimesh.sample.sample_surface_even(neural_mesh, cloud_size)
    q, _ = trimesh.sample.sample_surface_even(gt_mesh, cloud_size)

    # Compute pairwise L1 distances
    dists_pq = np.abs(p[:, np.newaxis, :] - q[np.newaxis, :, :]).sum(axis=2)  # Shape (cloud_size, cloud_size)
    dists_qp = np.abs(q[:, np.newaxis, :] - p[np.newaxis, :, :]).sum(axis=2)  # Shape (cloud_size, cloud_size)

    # Get the minimum distances
    sum_1 = dists_pq.min(axis=1).sum()  # Minimum for each p -> q
    sum_2 = dists_qp.min(axis=1).sum()  # Minimum for each q -> p

    # Compute Chamfer distance
    chamfer = 0.5 * (sum_1 + sum_2) / cloud_size
    return chamfer 

def sdf_capped_torus(X,Y,Z):
    sc = np.array([0.8, 0.6])       # Cap shape
    ra = 0.6                        # Major radius
    rb = 0.2     
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    sc = np.asarray(sc, dtype=np.float32)
    
    # Use the absolute value of X as per the formula.
    X_abs = np.abs(X)
    
    # Conditional computation for k:
    # If sc_y * |X| > sc_x * Y, compute the dot product; otherwise, use the norm of (X, Y)
    cond = (sc[1] * X_abs > sc[0] * Y)
    k = np.where(cond, X_abs * sc[0] + Y * sc[1], np.sqrt(X**2 + Y**2))
    
    # Compute the signed distance.
    dist = np.sqrt(X**2 + Y**2 + Z**2 + ra**2 - 2.0 * ra * k) - rb

    return dist

def bean_vizs(net, current_path):
    res = 60
    x_range = np.linspace(-1, 1., res)
    y_range = np.linspace(-0.1, 1., res)
    z_range = 0.*np.ones_like(y_range)
    X,Y,Z = np.meshgrid(x_range, y_range, z_range)
    eval_points =  np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    gt_dist = sdf_capped_torus(X,Y,Z)

    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]
    dz = 0
    grad_x = (np.roll(gt_dist, -1, axis=1) - np.roll(gt_dist, 1, axis=1)) / (2 * dx)
    grad_y = (np.roll(gt_dist, -1, axis=0) - np.roll(gt_dist, 1, axis=0)) / (2 * dy)

    skip = 2 # reduce the number of arrows for clarity
    fig, ax = plt.subplots()

    eval_points = tens(eval_points)
    gradients = gradient(net(eval_points), eval_points).detach().cpu().numpy()
    neural_grad_x = gradients[:,0].reshape((res,res,res))
    neural_grad_y = gradients[:,1].reshape((res,res,res))
    contour = ax.contourf(X[:,:,0], Y[:,:,0], gt_dist[:,:,0], levels=100, cmap='viridis')

    fig.colorbar(contour, label='Signed Distance')
    ax.quiver(X[::skip, ::skip,0], Y[::skip, ::skip,0],
            neural_grad_x[::skip, ::skip,0], neural_grad_y[::skip, ::skip,0],
            color='red', scale=50, width=0.005)
    fig.savefig(current_path +"neuralgradient.png", transparent=None)

    fig, ax = plt.subplots()
    contour = ax.contourf(X[:,:,0], Y[:,:,0], gt_dist[:,:,0], levels=100, cmap='viridis')
    plt.grid(visible=True)
    fig.colorbar(contour, label='Signed Distance')
    ax.quiver(X[::skip, ::skip,0], Y[::skip, ::skip,0],
            grad_x[::skip, ::skip,0], grad_y[::skip, ::skip,0],
            color='white', scale=50, width=0.005)
    
    fig.savefig(current_path+"sdfgradient.png", transparent=None)
    plt.rcParams['svg.fonttype'] = 'none' # Use text as text in SVG
    plt.savefig(current_path + "/sdfgradient" +'.svg', format = 'svg')

