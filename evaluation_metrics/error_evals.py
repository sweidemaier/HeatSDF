import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys

print(os.getcwd())
from evaluation_metrics import error_utils
sys.path.insert(0, os.getcwd())

import torch
import trimesh
from trainers.utils.new_utils import tens
from utils import load_imf
from trainers.utils.diff_ops import gradient
import numpy as np
from skimage.draw import ellipsoid
import csv
bs = 10000
global_eval = False
width_n = 0.01
width_f = 0.1
def eval(current_path, fine_mc_mesh, object_type):
    if(object_type == "lightbulb"):
        mesh_path = "/home/weidemaier/PDE Net/NFGP/meshes/lightbulb_centered_mesh.obj" 
        filename = "/home/weidemaier/PDE Net/NFGP/meshes/lightbulb_center.csv" 
        near_band_path = "/home/weidemaier/PDE Net/near_band_lightbulb.csv"
        far_band_path = "/home/weidemaier/PDE Net/far_band_lightbulb.csv"
    if(object_type == "armadillo"):
        mesh_path = "/home/weidemaier/PDE Net/arm_centered.obj" 
        filename = "/home/weidemaier/PDE Net/NFGP/armadillo2.csv" 
        near_band_path = "/home/weidemaier/PDE Net/near_band_armadillo.csv"
        far_band_path = "/home/weidemaier/PDE Net/far_band_armadillo.csv"
    if(object_type == "head"):
        mesh_path = "/home/weidemaier/PDE Net/fixed_head_centered_mesh.obj"  
        filename = "/home/weidemaier/PDE Net/NFGP/meshes/SIRENhead.csv" 
        near_band_path = "/home/weidemaier/PDE Net/near_band_head.csv"
        far_band_path = "/home/weidemaier/PDE Net/far_band_head.csv"
    if(object_type == "hand"):
        mesh_path = "/home/weidemaier/PDE Net/NFGP/meshes/hand_centered_mesh.obj" 
        filename = "/home/weidemaier/PDE Net/NFGP/meshes/hand_center.csv"
        near_band_path = "/home/weidemaier/PDE Net/near_band_hand.csv"
        far_band_path = "/home/weidemaier/PDE Net/far_band_hand.csv"
    if(object_type == "bunny"):
        mesh_path = "/home/weidemaier/PDE Net/NFGP/meshes/bunny_centered_mesh.obj" 
        filename = "/home/weidemaier/PDE Net/NFGP/meshes/bunny_center.csv"
        near_band_path = "/home/weidemaier/PDE Net/near_band_bunny.csv"
        far_band_path = "/home/weidemaier/PDE Net/far_band_bunny.csv"
    if(object_type == "bean"):
        mesh_path = "/home/weidemaier/PDE Net/NFGP/meshes/cappedTorus.obj"
        near_band_path = "/home/weidemaier/PDE Net/bean_near.csv"
        far_band_path = "/home/weidemaier/PDE Net/bean_far.csv"
    ### load neural SDF approx
    net, cfg = load_imf(
        "/home/weidemaier/PDE Net/" + current_path, 
        return_cfg=False)
    mesh = trimesh.load(mesh_path, force='mesh')
    #center_mesh = error_utils.center_mesh("/home/weidemaier/PDE Net/NFGP/meshes/dog.obj")
    #trimesh.exchange.export.export_mesh(center_mesh, "/home/weidemaier/PDE Net/NFGP/meshes/dog_center.obj", file_type=None, resolver=None) #"+ str(i) + ".
    #pts = error_utils.loading_pts("/home/weidemaier/PDE Net/NFGP/bucky_uniform.csv", center = True)
    #np.savetxt(current_path + "bucky_uniform_center.csv", pts, delimiter = ",", header = "x,y,z")
        


    ### Surface errors
    err_L2 = error_utils.L2_loss(mesh, net)
    chamfer = error_utils.chamfer_new_score(fine_mc_mesh, mesh, cloud_size=10000)

    
    ### Eikonal errors
    band_near= tens(error_utils.loading_pts(near_band_path, False))
    #np.savetxt("bean_near.csv", band_near.detach().cpu() , delimiter = ",", header = "x,y,z")
    band_far= tens(error_utils.loading_pts(far_band_path, False)) 
    #np.savetxt("bean_far.csv", band_far.detach().cpu() , delimiter = ",", header = "x,y,z")
    bs = 10000
    xyz = tens(np.random.uniform(-1.2, 1.2, (bs, 3)))
    norm_global = torch.norm(gradient(net(xyz), xyz), dim = -1)
    global_median = torch.abs(norm_global - torch.ones(bs,1).cuda()).median()
    norm_global = torch.abs(norm_global - torch.ones(bs,1).cuda()).mean()
    err_W12_near = torch.abs(torch.norm(gradient(net(band_near), band_near), dim = -1) - torch.ones(band_near.shape[0]).cuda()).mean() 
    err_W12_median = torch.abs(torch.norm(gradient(net(band_near), band_near), dim = -1) - torch.ones(band_near.shape[0]).cuda()).median()
    err_W12_far = torch.abs(torch.norm(gradient(net(band_far), band_far), dim = -1) - torch.ones(band_far.shape[0]).cuda()).mean() 
    err_W12_far_median = torch.abs(torch.norm(gradient(net(band_far), band_far), dim = -1) - torch.ones(band_far.shape[0]).cuda()).median() 
    error_utils.Eikonal_hist(current_path, object_type, net, band_far)
    ### Normals alignement
    points = mesh.triangles_center
    input_normals = mesh.face_normals
    face_centers = tens(points)
    grad = gradient(net(face_centers), face_centers)
    neural_normals = gradient(net(face_centers), face_centers) /torch.norm(grad, dim = -1).view(points.shape[0], 1) 
    cos = (1 - torch.sum(neural_normals * tens(input_normals), dim=1)).mean()
    
    #print(error_utils.get_gt(mesh_path, band_far.detach().cpu().numpy()))

    gt = error_utils.get_gt(mesh_path,band_far.detach().cpu().numpy())#sdf_capped_torus(band_far.detach().cpu().numpy()[:,0], band_far.detach().cpu().numpy()[:,1], band_far.detach().cpu().numpy()[:,2])
    gt_global = error_utils.sdf_capped_torus(xyz.detach().cpu().numpy()[:,0], xyz.detach().cpu().numpy()[:,1], xyz.detach().cpu().numpy()[:,2])
    np.savetxt(current_path + "gt_sdf_band.csv", gt , delimiter = ",", header = "x,y,z")
    np.savetxt(current_path + "global_pts.csv", xyz.detach().cpu() , delimiter = ",", header = "x,y,z")
    np.savetxt(current_path + "gt_sdf_global.csv", gt_global , delimiter = ",", header = "x,y,z")
    ### SDF error
    band_far.requires_grad = False
    neural_dist = net(band_far).reshape(band_far.shape[0])
    gt = error_utils.get_gt(mesh_path, band_far.detach().cpu().numpy())
    print(gt.shape)
    print(neural_dist.shape)
    sdf_diff = np.abs(gt - neural_dist.detach().cpu().numpy()).mean()
    print((np.abs(gt - neural_dist.detach().cpu().numpy())).shape)
    
    torch.cuda.empty_cache()
    
    
    if (object_type == "bean"):
        x,y = error_utils.SDF_scatter(current_path, mesh_path, net, band_far, bean = True)
        vec = np.stack((x, y), axis=1)
        vec = vec.reshape(x.shape[0],2)
        print(vec)
        np.savetxt(current_path + "bean_scatter.csv", vec, delimiter = ",", header = "x,y")
        error_utils.bean_vizs(net, current_path)
    else:
        x,y = error_utils.SDF_scatter(current_path, mesh_path, net, band_far)
        vec = np.stack((x, y), axis=1)
        vec = vec.reshape(x.shape[0],2)
        print(vec)
        np.savetxt(current_path + "/hand_scatter.csv", vec, delimiter = ",", header = "x,y")
    #print(cfg.input.parameters.param1)
    #print(cfg.input.parameters.param2)
    print("On surf", err_L2)
    print("Chamfer", chamfer)
    print("Eikonal - surface:", err_W12_near)
    print("Eikonal - surface - median", err_W12_median)
    print("Eikonal - narrow", err_W12_far)
    print("Eikonal - narrow - median", err_W12_far_median)
    print("Eikonal global - median", norm_global)
    print("Eikonal global - median", global_median)
    print("Normal alignement:", cos)
    print("SDF error:", sdf_diff)
    return err_L2, chamfer, err_W12_near, err_W12_median, err_W12_far, err_W12_far_median, norm_global, global_median, cos, sdf_diff
    '''
    





    

    import matplotlib.pyplot as plt
    
    bs = band_far.shape[0]
    ### difference to groundtruth SDF
    neural_dist = net(band_far)
    gt = error_utils.get_gt(mesh_path, band_far.detach().cpu().numpy())
    np.savetxt(current_path + "gt_sdf_band.csv", gt , delimiter = ",", header = "x,y,z")
    gt_res = np.reshape(gt, (bs,1))

    diff = np.reshape(gt_res - neural_dist.detach().cpu().numpy(), bs)
    sdf_error = np.abs(diff).mean()
    fig, ax = plt.subplots()
    print("SDF:", sdf_error)
    ax.scatter(np.abs(gt),diff , s=0.1)
    fig.savefig(current_path +"SDF_scatter.png", transparent=None)

    bs = 10000
    xyz = np.random.uniform(-1.1, 1.1, (bs, 3))
    np.savetxt(current_path + "global_pts.csv", xyz , delimiter = ",", header = "x,y,z")
   
    far_band= tens(xyz) #tens(error_utils.loading_pts(far_band_path)) 
    bs = far_band.shape[0]
    ### difference to groundtruth SDF
    neural_dist = net(far_band)
    gt = error_utils.get_gt(mesh_path, far_band.detach().cpu().numpy())
    np.savetxt(current_path + "gt_sdf_global.csv", gt , delimiter = ",", header = "x,y,z")
    gt_res = np.reshape(gt, (bs,1))
    eikonal = torch.norm(gradient(net(far_band), far_band), dim = -1) - torch.ones_like(torch.norm(gradient(net(far_band), far_band), dim = -1)).view(bs)
    
    fig2, ax2 = plt.subplots()

    ax2.hist2d(gt,eikonal.detach().cpu().numpy() , bins=(np.arange(-0.25, 1.25, 0.05), np.arange(-1., 1, 0.05)))
    fig2.savefig(current_path + "eikonal_hist.png", transparent=None)
    plt.show()
    return err_L2.item(), champher, err_W12_near.item(),err_W12_median.item(), err_W12_far.item(), err_W12_far_median.item(), norm_global.item(), global_median.item(), normal_alignement.item(), sdf_error'''