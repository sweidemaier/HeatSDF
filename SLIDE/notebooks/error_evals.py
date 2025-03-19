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
def eval(current_path):
    #new_net_path = "/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs_2025-Feb-13-16-05-38" #"/home/weidemaier/PDE Net/NFGP/logs/fine_head"
    mesh_path = "/home/weidemaier/PDE Net/NFGP/headA_centered.obj"
    filename = "/home/weidemaier/PDE Net/NFGP/head_fixed.csv" #headA_org.csv"
    
    ### load neural SDF approx
    net, cfg = load_imf(
        "/home/weidemaier/PDE Net/" + current_path, 
        return_cfg=False
    #,ckpt_fpath = new_net_path + "/checkpoints/epoch_1_iters_2000.pt"
        )
    #mesh_path = cfg.eval_props.centermesh
    #filename = cfg.eval_props.ptfile
    '''optional: 
    mesh_path = cfg....
    filename = cfg ...
    '''
    ### load groundtruth SDF
    #error_utils.center_mesh(mesh_path)
    #vec, _ = error_utils.gt_band(mesh_path, 10000,0.1)
    #np.savetxt("gt_band_head.csv", vec.detach().cpu() , delimiter = ",", header = "x,y,z")
    ### point samples
    on_surf_np = error_utils.loading_pts(filename)
    on_surf = tens(on_surf_np)
    band_near, bs_near = error_utils.gt_band(mesh_path, 10, width_n)
    band_far, bs_far = error_utils.gt_band(mesh_path, 10, width_f)
    np.savetxt("head_band.csv", band_near.detach().cpu() , delimiter = ",", header = "x,y,z")
    ### error L_inf (on surface)
    err_L_inf = torch.max(torch.abs(net(on_surf)))
    ### error L_inf (on surface)
    err_L2 = torch.square(net(on_surf)).mean()
    ### error W1,2
    xyz = tens(np.random.uniform(-1, 1, (bs, 3)))
    norm_global = torch.norm(gradient(net(xyz), xyz), dim = -1)
    print(norm_global)
    err_W12_near = torch.square(torch.norm(gradient(net(band_near), band_near), dim = -1) - torch.ones_like(torch.norm(gradient(net(band_near), band_near), dim = -1))).mean() 
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
    print("Eikonal_far", err_W12_far)
    #print("gt_diff:", diff)
    #print("normals:", normal_alignement)
    return err_L2.item(), err_W12_near.item(), err_W12_far.item()#, diff