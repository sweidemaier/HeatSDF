import os
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys

print(os.getcwd())
from notebooks import error_utils
#os.chdir("..")

sys.path.insert(0, os.getcwd())
import torch
import trimesh
from trainers.utils.new_utils import tens
from trainers.utils.vis_utils import imf2mesh
from trainers import RHS, analyticSDFs
from utils import load_imf, write_obj
from trainers.utils.diff_ops import gradient
import numpy as np
from skimage.draw import ellipsoid
import csv
import mesh2sdf
bs = 10000
global_eval = False
width_n = 0.01
width_f = 0.1
def eval(current_path, fine_mc_mesh):
    #new_net_path = "/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs_2025-Feb-13-16-05-38" #"/home/weidemaier/PDE Net/NFGP/logs/fine_head"
    mesh_path = "/home/weidemaier/PDE Net/NFGP/meshes/fine_head_centered_mesh.obj" #"/home/weidemaier/PDE Net/NFGP/meshes/beathoven_centered_mesh.obj" #"/home/weidemaier/PDE Net/fine_head_centered_mesh.obj" # 
    filename = "/home/weidemaier/PDE Net/NFGP/head_fixed.csv" #headA_org.csv"
    near_band_path = "/home/weidemaier/PDE Net/near_band_head.csv"
    far_band_path = "/home/weidemaier/PDE Net/far_band_head.csv"
    ### load neural SDF approx
    net, cfg = load_imf(
        "/home/weidemaier/PDE Net/" + current_path, 
        return_cfg=False)
    
    #error_utils.center_mesh(mesh_path)
    ### load groundtruth SDF
    #error_utils.center_mesh(mesh_path)
    #print("safed center ")
    #vec_far, _ = error_utils.gt_band(mesh_path, 10000,0.1, 20000)
    #vec_near, _ = error_utils.gt_band(mesh_path, 10000,0.01, 50000)
    #np.savetxt("near_band_lightbulb.csv", vec_near.detach().cpu() , delimiter = ",", header = "x,y,z")
    #np.savetxt("far_band_lightbulb.csv", vec_far.detach().cpu() , delimiter = ",", header = "x,y,z")
    ### point samples
    on_surf_np = error_utils.loading_pts(filename, center = True)
    champher = error_utils.chamfer_score(on_surf_np, fine_mc_mesh.vertices, cloud_size=5000)
    print(champher)
    on_surf = tens(on_surf_np)
    band_near= tens(error_utils.loading_pts(near_band_path)) #band(net, 10000, 0.01)#gt_band(mesh_path, 10, width_n)
    band_far= tens(error_utils.loading_pts(far_band_path)) #band(net, 10000, 0.1) #gt_band(mesh_path, 10, width_f)
    
    #np.savetxt("head_band.csv", band_near.detach().cpu() , delimiter = ",", header = "x,y,z")
    ### error L_inf (on surface)
    err_L_inf = torch.max(torch.abs(net(on_surf)))
    ### error L_inf (on surface)
    err_L2 = torch.square(net(on_surf)).mean()
    ### error W1,2
    xyz = tens(np.random.uniform(-1.2, 1.2, (bs, 3)))
    norm_global = torch.norm(gradient(net(xyz), xyz), dim = -1)
    norm_global = torch.abs(norm_global - torch.ones(bs,1).cuda()).mean()
    err_W12_near = torch.abs(torch.norm(gradient(net(band_near), band_near), dim = -1) - torch.ones_like(torch.norm(gradient(net(band_near), band_near), dim = -1))).mean() 
    err_W12_median = torch.abs(torch.norm(gradient(net(band_near), band_near), dim = -1) - torch.ones_like(torch.norm(gradient(net(band_near), band_near), dim = -1))).median()

    err_W12_far = torch.abs(torch.norm(gradient(net(band_far), band_far), dim = -1) - torch.ones_like(torch.norm(gradient(net(band_far), band_far), dim = -1))).mean() 
    ### normal alignement
    #input_normals, points = error_utils.get_normals(mesh_path)
    #face_centers = tens(points)
    #normal_alignement = torch.norm(gradient(net(face_centers), face_centers) + tens(input_normals), dim = -1).mean()
    ### difference to groundtruth SDF
    #gt = error_utils.get_gt(mesh_path, band_near.detach().cpu().numpy())
    #diff = torch.abs(tens(gt) + net(band_near)).mean()
    
    print(cfg.input.parameters.param1)
    print(cfg.input.parameters.param2)
    print("On surf", err_L2)
    print("Eikonal", err_W12_near)
    print("Eikonal - median", err_W12_median)
    print("Eikonal_far", err_W12_far)
    print("Eikonal_global", norm_global)
    #print("gt_diff:", diff)
    #print("normals:", normal_alignement)
    return err_L2.item(), champher, err_W12_near.item(),err_W12_median.item(), err_W12_far.item(), norm_global.item()#, diff