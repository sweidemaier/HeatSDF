import numpy as np
import trimesh
from trainers.utils.new_utils import tens
def val_sphere(bs):
    r = 1
    phi = np.linspace(0, 2*np.pi, bs)
    theta = np.linspace(0, np.pi, bs)
    i = 0
    xy = [None]*bs
    xy_org = [None]*bs
    while i < bs:
        xy[i] = [np.sin(theta[i])*np.cos(phi[i]), np.sin(theta[i])*np.sin(phi[i]), np.cos(theta[i])]
        xy_org[i] = [r, phi[i], theta[i]]
        i += 1
    return xy, xy_org

def uni_sphere(bs):
    x = np.random.standard_normal(bs)
    y = np.random.standard_normal(bs)
    z = np.random.standard_normal(bs)
    i = 0
    xyz = [None]*bs
    while i < bs:
        normi = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        xyz[i] = [x[i]/normi, y[i]/normi, z[i]/normi]
        i += 1
    return xyz, xyz
def fibonacci_sphere(samples):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    t = np.random.uniform(0, 2*np.pi)
    Ry = np.matrix([[np.cos(t), 0, np.sin(t)],[0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    Rx = np.matrix([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    R = Ry*Rx
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        vec = R.dot([x,y,z])
 
        points.append(vec)
    
    
    return points, points
def val_uni_sphere(bs):
    x = np.linspace(-2, 2, bs)
    y = np.linspace(-2, 2, bs)
    z = np.linspace(-2, 2, bs)
    i = 0
    xyz = [None]*bs
    while i < bs:
        normi = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        xyz[i] = [x[i]/normi, y[i]/normi, z[i]/normi]
        i += 1
    return xyz, xyz

def plane(bs):
    xy_sample = np.random.uniform(-2,2, (bs, 2))
    i = 0
    xy = [None]*bs
    while i < bs:
        xy[i] = [xy_sample[i,0], xy_sample[i,1], 0]
        i += 1
    return xy

def torus(bs):
    i = 0
    r_1 = 0.2
    r_2 = 0.09
    phi = np.random.uniform(0, 2*np.pi, bs)
    theta = np.random.uniform(0, 2*np.pi, bs)
    xy = [None]*bs
    while i < bs:
        xy[i] = [(r_2*np.cos(theta[i]) + r_1)*np.cos(phi[i]), (r_2*np.cos(theta[i]) + r_1)*np.sin(phi[i]), r_2*np.sin(theta[i])]
        i += 1
    return xy

def mesh(path):
    mesh = trimesh.load(path)
    res = mesh.vertices

def compFEM(bs):
    xyz_org, xy = fibonacci_sphere(bs)
    xyz_org = np.reshape(xyz_org, (bs, 3))
    xy = np.reshape(xy, (bs, 3))
    i = 0
    xyz = [None]*bs
    while i < bs:
        xyz[i] = [2*xyz_org[i][0], xyz_org[i][1], 0.5*xyz_org[i][2]*(1+0.5*np.sin(2*np.pi*xyz_org[i][0]))]
        i += 1
    return xyz, xyz_org

