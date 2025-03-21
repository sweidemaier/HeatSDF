import scipy.spatial as tree
import scipy.integrate as integral
import numpy as np
import time

def bump_func(x):
        if (abs(x) > 1):
            return 0
        else:
            return np.exp(1/((abs(x)**2)-1))   
    
def comp_weights(pointcloud, epsilon, dim = 2 ):
    start = time.time()
    w = np.zeros(np.shape(pointcloud)[0])
    c = integral.quad(bump_func, -1, 1)
    
    c_eps = c[0]*(epsilon**dim)
    #c_eps = (epsilon**dim)
    tr = tree.cKDTree(pointcloud)
    
    r = epsilon
    print("initial eps:", r)
    p = tr.query_ball_point(x = pointcloud, r = r, workers = -1)
    while any(len(ball) < 12 for ball in p):
        r *= 2

        p = tr.query_ball_point(x = pointcloud, r = r, workers = -1)
    
    j = 0
    print("scaled eps:",r)
    while j < np.size(p):
        ball_indices = p[j]
        ball_points = pointcloud[ball_indices]
        dists = np.linalg.norm(pointcloud[j]-ball_points, axis = 1)
        sum = np.sum([bump_func(dists[i]/r) for i in range(len(dists))])
        w[j] = c_eps/sum

        j += 1
    
    w = w/np.sum(w)
    print("Total computation time:", time.time() - start)
    return w


