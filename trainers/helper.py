import scipy.spatial as tree
import scipy.integrate as integral
import numpy as np

def bump_func(x):
        if (abs(x) > 1):
            return 0
        else:
            return np.exp(1/((abs(x)**2)-1))   
    
def comp_weights(pointcloud, epsilon, dim = 2 ):
    w = np.zeros(np.shape(pointcloud)[0])
    c = integral.quad(bump_func, -1, 1)
    
    #c_eps = c[0]*(epsilon**dim)
    c_eps = (epsilon**dim)
    tr = tree.KDTree(pointcloud)
    i = 0
    eps = epsilon
    print(eps)
    p = tr.query_ball_point(x = pointcloud, r = eps, workers = -1)
    while i <  np.size(p):
        if np.size(p[i]) < 12:
            eps *= 2
            p = tr.query_ball_point(x = pointcloud, r = eps, workers = -1)
            i = 0
            print(eps)
        i = i+1
    print(np.size(p[:]))
    i = 0
    sum_2 = 0
    while i < np.size(p):
        count = np.size(p[i])
        j = 0
        sum = 0
        while j < count:
            p_j = pointcloud[p[i][j]]
            diff = np.subtract(pointcloud[i],p_j)
            if(np.linalg.norm(diff, 2)> eps): print("wrong calc:", np.linalg.norm(diff, 2))# return 
            sum += bump_func(np.linalg.norm(diff, 2)/eps)
            j += 1
        #print(sum)
        w[i] = c_eps / sum 
        #print(w[i])
        sum_2 += w[i]
        
        #print(w[i])
        i = i + 1
        #if(i%1000 == 0): print(np.size(p[i+250]), i) #print(i)
    w = w/sum_2
    print(np.sum(w))
    print(w)
    return w


