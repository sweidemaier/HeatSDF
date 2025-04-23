import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import torch
import trimesh
from trainers.utils.new_utils import tens
from trainers.utils.vis_utils import imf2mesh
from evaluation_metrics import error_evals
from utils import load_imf
from trainers.utils.diff_ops import gradient
import numpy as np
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
new_net_path = "/home/weidemaier/HeatSDF/logs/"

stri = ("SDF2025-Apr-23-10-13-10/SDF_step", "")


for i in range(len(stri)):
    net, cfg = load_imf(

        new_net_path+stri[i], 
        return_cfg=False
        #,ckpt_fpath = new_net_path + stri[i] + "/checkpoints/epoch_36_iters_37000.pt"
    )
    mesh = imf2mesh(net, res = 256, normalize=True, bound = 1.15, threshold=0.0)
    trimesh.exchange.export.export_mesh(mesh, new_net_path + str(stri[i]) + "/vis" + ".obj", file_type=None, resolver=None) 

    L2, chamfer, near, med, far, far_med, glob,glob_med, normal_al, sdf = error_evals.eval("logs/" + stri[i], mesh, "hand")
    
    with open(new_net_path  + "overview.txt", "a") as f:
        f.write(str(cfg.input.parameters.param1) +" / " + str(cfg.input.parameters.param2)+" & "+str(f"{L2:.4g}")+" / "+str(f"{chamfer:.4g}")  +" & "+str(f"{near:.4g}")+" / "+str(f"{med:.4g}") +" / "+ str(f"{far:.4g}")+" / "+ str(f"{far_med:.4g}")+" / "+ str(f"{glob:.4g}")+" / "+ str(f"{glob_med:.4g}")+" & "+str(f"{normal_al:.4g}")+str(f"{sdf:.4g}") +"\\\ "+" % "+ stri[i])

        f.write("\n")

    f = open(new_net_path + "overview.txt","r")
    torch.cuda.empty_cache()