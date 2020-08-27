
import numpy as np


def compute_barycenter(points):
    return np.mean(points,axis=0)

def sample_point(n_points, sampling='uniform'):
    if (sampling=='uniform'):
        return np.random.randint(0,n_points)
    
def take_a_step(point,fixedpoint,epsilon=0.01):
    diff_vector = fixedpoint - point
    pprime = point + epsilon*diff_vector
    return pprime

def check_convergence(points,fixedpoint,epsilon=0.01,norm='l2'):
    if(norm=='l2'):
        dist = np.linalg.norm(points-fixedpoint,axis=1)
        return np.all(dist<epsilon)
    
def map_back(ranking):
    M = ranking.shape[0]
    P = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            P[i,j] = ranking[i] / (ranking[i]+ranking[j])
    return P