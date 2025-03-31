import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import trimesh
from trainers.utils.new_utils import tens
import numpy as np
import csv
import mesh2sdf
import mesh_to_sdf
from utils import load_imf, write_obj

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
    print(pts)
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
    
    write_obj("bunny_centered_mesh.obj", points, mesh.faces)
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
def chamfer_score(points_neural, points_gt, cloud_size = 5000):
    ind_neural = np.random.choice(points_neural.shape[0], cloud_size)
    ind_gt = np.random.choice(points_gt.shape[0], cloud_size)
    p = points_neural[ind_neural]
    q = points_gt[ind_gt]
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
    