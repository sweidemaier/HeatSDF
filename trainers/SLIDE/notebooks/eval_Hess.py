import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import torch
import trimesh
from trainers.utils.new_utils import tens
from trainers.utils.vis_utils import imf2mesh
from notebooks import error_evals
from trainers import RHS, analyticSDFs
from utils import load_imf, write_obj
from trainers.utils.diff_ops import gradient
import numpy as np
new_net_path = "/home/weidemaier/Hessian/surface_reconstruction/log/sdf/headA_centered"
net, cfg = load_imf(
        new_net_path, 
        return_cfg=False
        #,ckpt_fpath = new_net_path# + stri[i]
    )
print(net)