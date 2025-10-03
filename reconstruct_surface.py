import os
import trimesh
from trainers.standard_utils import load_imf
from trainers.utils.vis_utils import imf2mesh

base_dir = os.path.dirname(os.path.abspath(__file__))


### file to load neural SDF and extract isosurface

###your network path; as an example we use the path to our SDF-initialization-network
path = os.path.join(base_dir, "configs", "initialization_network")
### load network
net, cfg = load_imf(path)
### run marching cubes
mesh = imf2mesh(net, res = 256, normalize=True, bound = 1.15, threshold=0.0)
### safe to main folder
trimesh.exchange.export.export_mesh(mesh, path + "/visualization" + ".obj", file_type=None, resolver=None)
