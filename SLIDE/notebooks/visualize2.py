import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.measure import marching_cubes
from utils import load_imf
from trainers.utils.new_utils import tens
from trainers.utils.diff_ops import gradient
new_net_path = "/home/weidemaier/PDE Net/logs/new looped heads/NeuralSDFs2_2025-Feb-28-00-49-20"

net, cfg = load_imf(
        new_net_path, 
        return_cfg=False
        #,ckpt_fpath = new_net_path + stri[i] + "/checkpoints/epoch_10_iters_11000.pt"
    )
def render_sdf_quad(render_path, contour_path, gradient_path,  P0, P1, P2, res=100):
    """
    Renders a 3D sdf along a specified (planar) quad in space.
    
    Args:
        render_path (str): Path to export a color render
        contour_path (str): Path to export the contour plot
        gradient_path (str): Path to export the gradient norm plot
        model : SDF neural model
        P0 (array): First point of the quad ( (0,0) in parameter space)
        P1 (array): Second point of the quad ( (1,0) in parameter space)
        P2 (array): Thirs point of the quad ( (0,1) in parameter space)
        device (str): cpu or cuda
        res (int, optional): Image resolution. Defaults to 800.
        batch_size (int, optional): Size of forward batches. Defaults to 1000.
    """

    dx = P1 - P0
    dy = P2 - P0
    X = np.linspace(0,1, res)
    #resY = round(res * M.geometry.norm(dy)/M.geometry.norm(dx))
    Y = np.linspace(0,1, res)

    pts = []
    for ax in X:
        for ay in Y:
            p = P0 + ax*dx + ay*dy
            pts.append(p)
    pts = np.array(pts)
    print(pts.shape)
    
    dist_values = net(tens(pts)).view(res**2) #(model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.detach().cpu().numpy()
    print(dist_values)
    img = dist_values.reshape((res,res)).T
    img = img[::-1,:]
    print(np.amin(img))
    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1
    print(vmin, vmax)
    vmin = -0.3
    vmax = 0.3
    p = tens(pts)
    grad_values = gradient(net(p),p )
    grad_values = grad_values.detach().cpu().numpy()

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm)
        plt.axis('off')
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        #plt.contour(img, levels=24, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.1, 0.15, 0.2, 0.25, 0.3], colors='k', linestyles="solid", linewidths=0.3)
        #plt.contour(img, levels=[-0.1], colors='k', linestyles="solid", linewidths=0.3)
    
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,res)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr")
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)

render_sdf_quad(new_net_path + "1", new_net_path+ "2", new_net_path+ "3", np.asarray([-1.2,-1.2,0.1]), np.asarray([-1.2,1.2,0.1]), np.asarray([1.2,-1.2,0.1]))
