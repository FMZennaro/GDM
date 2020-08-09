
# AGGREGATION LIBRARY
# [MxMxE] -> [MxE]
# [MxE] -> [M]


import numpy as np

def WA_3D(tensor,w=None):
    # [MxMxE] -> [MxM]
    return np.average(tensor,axis=2,weights=w)

def OWA_3D(tensor,w=None):
    # [MxMxE] -> [MxM]
    matrix = np.zeros((tensor.shape[0],tensor.shape[1]))
    for i in range(0,tensor.shape[0]):
        for j in range(0,tensor.shape[1]):
            n = np.average(np.sort(tensor[i,j,:])[::-1],weights=w)
            matrix[i,j] = n
    return matrix

def WA_2D(matrix,w=None):
    # [MxE] -> [M]
    return np.average(matrix,axis=1,weights=w)

def OWA_2D(matrix,w=None):
    # [MxE] -> [M]
    vector = np.zeros(matrix.shape[0])
    for i in range(0,matrix.shape[0]):
        n = np.average(np.sort(matrix[i,:]),weights=w)
        vector[i] = n
    return vector

def NF_2D(matrix):
    # [MxE] -> [M]
    return (np.sum(matrix,axis=1) - np.diag(matrix)) - matrix.shape[1] + 1

    
    


