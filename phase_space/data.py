
import numpy as np

def datapoints1():
    D = np.array([[0,0],
                 [0,1],
                 [1,0]], dtype='float32')
    return D

def randomdata(n_choices,n_experts):
    
    D = np.zeros((n_choices, n_choices,n_experts))

    for e in range(n_experts):
        M = np.random.uniform(size=(n_choices,n_choices))

        Dl = np.tril(M,-1)
        Du = np.tril((1-M),-1)

        D[:,:,e] = Dl + Du.T + np.diag([.5]*n_choices)
        
    return D