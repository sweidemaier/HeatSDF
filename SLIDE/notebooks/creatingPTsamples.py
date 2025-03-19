import os
import sys
os.chdir("..")
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import torch
import trimesh
from trainers.utils.new_utils import tens
from trainers.utils.vis_utils import imf2mesh
from trainers import RHS, analyticSDFs
from utils import load_imf, write_obj
from trainers.utils.diff_ops import gradient
import numpy as np
import trimesh

bs = 200000
path = "/home/weidemaier/New Network/NFGP/bunny_curvs.obj"

mesh = trimesh.load_mesh(path, "obj")
print(mesh.vertices)
verts = mesh.vertices #trimesh.sample.sample_surface_even(mesh, bs)[0]
#mesh.vertices #t
'''
def fibonacci_sphere(samples):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    t = np.random.uniform(0, 2*np.pi)
    #Ry = np.matrix([[np.cos(t), 0, np.sin(t)],[0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    #Rx = np.matrix([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    #R = Ry*Rx
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        vec = [x,y,z]
 
        points.append(vec)
    return points
vec_out = fibonacci_sphere(100000)'''

#np.savetxt("bunny.csv", verts , delimiter = ",", header = "x,y,z")

###

filename = "/home/weidemaier/PDE Net/NFGP/beethovencomplete.xyz"
xyz = open(filename, "r")
line = xyz.readline()

with open(filename) as f:
    for i, _ in enumerate(f):
        pass
i 

vec = [None]*i
i = 0
for line in xyz:
    x, y, z = line.split()

    vec[i] = [np.float32(x),np.float32(y),np.float32(z)]
    i += 1

# Calculate what you need here

xyz.close()
np.savetxt("beathoven.csv", vec , delimiter = ",", header = "x,y,z")
