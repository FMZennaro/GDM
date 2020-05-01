
import numpy as np

def get_PREF_tensor_example1():
    PREF_tensor = np.zeros((3,3,4))
    
    PREF_tensor[:,:,0]= np.array([[.5,.7,.3],
                 [.3,.5,.4],
                 [.7,.6,.5]])

    PREF_tensor[:,:,1] = np.array([[.5,.42,.35],
                     [.58,.5,.8],
                     [.65,.2,.5]])
    
    PREF_tensor[:,:,2] = np.array([[.5,.53,.32],
                     [.47,.5,.79],
                     [.68,.21,.5]])
    
    PREF_tensor[:,:,3] = np.array([[.5,.56,.28],
                     [.44,.5,.3],
                     [.72,.7,.5]])
    
    return PREF_tensor

def get_PREF_consensus_matrix_Dopazo1():
    PREF_consensus_matrix= np.array([[.5,.2666,.3584, 0,675],
                         [.7334,.5,.6666,.725],
                         [.6416,.3333,.5,.5584],
                         [.325,.275,.4416,.5]])
    
    return PREF_consensus_matrix


def get_PREF_tensor_Viedma1():
    PREF_tensor = np.zeros((6,6,8))
    
    PREF_tensor[:,:,0]= np.array([[0.5, 0.4, 0.6, 0.9, 0.7, 0.8], 
                           [0.6, 0.5, 0.7, 1, 0.8, 0.9], 
                           [0.4, 0.3, 0.5, 0.8, 0.6, 0.7], 
                           [0.1, 0, 0.2, 0.5, 0.3, 0.4], 
                           [0.3, 0.2, 0.4, 0.7, 0.5, 0.6], 
                           [0.2, 0.1, 0.3, 0.6, 0.4, 0.5]])
    
    PREF_tensor[:,:,1]= np.array([[0.5, 0.7, 0.8, 0.6, 1, 0.9],
                           [0.3, 0.5, 0.6, 0.4, 0.8, 0.7],
                           [0.2, 0.4, 0.5, 0.3, 0.7, 0.6],
                           [0.4, 0.6, 0.7, 0.5, 0.9, 0.8],
                           [0, 0.2, 0.3, 0.1, 0.5, 0.4],
                           [0.1, 0.3, 0.4, 0.2, 0.6, 0.5]])
    
    PREF_tensor[:,:,2]= np.array([[0.5, 0.69, 0.12, 0.2, 0.36, 0.9],
                           [0.31, 0.5, 0.06, 0.1, 0.2, 0.8],
                           [0.88, 0.94, 0.5, 0.64, 0.8, 0.98],
                           [0.8, 0.9, 0.36, 0.5, 0.69, 0.97],
                           [0.64, 0.8, 0.2, 0.31, 0.5, 0.94],
                           [0.1, 0.2, 0.02, 0.03, 0.06, 0.5]])
    
    PREF_tensor[:,:,3]= np.array([[0.5, 0.1, 0.36, 0.69, 0.16, 0.26],
                           [0.9, 0.5, 0.84, 0.95, 0.62, 0.76],
                           [0.64, 0.16, 0.5, 0.8, 0.25, 0.39],
                           [0.31, 0.05, 0.2, 0.5, 0.08, 0.14],
                           [0.84, 0.38, 0.75, 0.92, 0.5, 0.66],
                           [0.74, 0.24, 0.61, 0.86, 0.34, 0.5]])
    
    PREF_tensor[:,:,4]= np.array([[0.5, 0.55, 0.45, 0.25, 0.7, 0.3], 
                           [0.45, 0.5, 0.7, 0.85, 0.4, 0.8], 
                           [0.55, 0.3, 0.5, 0.65, 0.7, 0.6], 
                           [0.75, 0.15, 0.35, 0.5, 0.95, 0.6], 
                           [0.3, 0.6, 0.3, 0.05, 0.5, 0.85], 
                           [0.7, 0.2, 0.2, 0.4, 0.15, 0.5]])
    
    PREF_tensor[:,:,5]= np.array([[0.5, 0.7, 0.75, 0.95, 0.6, 0.85], 
                           [0.3, 0.5, 0.55, 0.8, 0.4, 0.65], 
                           [0.25, 0.45, 0.5, 0.7, 0.6, 0.45], 
                           [0.05, 0.2, 0.3, 0.5, 0.85, 0.4], 
                           [0.4, 0.6, 0.4, 0.15, 0.5, 0.75], 
                           [0.15, 0.35, 0.55, 0.6, 0.25, 0.5]])
               
    PREF_tensor[:,:,6]= np.array([[0.5, 0.34, 0.25, 0.82, 0.75, 0.87],
                           [0.66, 0.5, 0.25, 0.18, 0.82, 0.91],
                           [0.75, 0.75, 0.5, 0.94, 0.91, 1],
                           [0.18, 0.82, 0.065, 0.5, 0.34, 0.75],
                           [0.25, 0.18, 0.09, 0.66, 0.5, 0.82],
                           [0.13, 0.09, 0, 0.25, 0.18, 0.5]])
               
    PREF_tensor[:,:,7]= np.array([[0.5, 0.13, 0.18, 0.34, 0.75, 0.09],
                           [0.87, 0.5, 0.66, 0.82, 0.91, 0.25],
                           [0.82, 0.34, 0.5, 0.75, 0.87, 0.82],
                           [0.66, 0.18, 0.25, 0.5, 0.75, 0.91],
                           [0.25, 0.09, 0.13, 0.25, 0.5, 0.97],
                           [0.91, 0.75, 0.18, 0.09, 0.03, 0.5]])
    
    return PREF_tensor

def get_W1_Viedma1():
    return np.array([0,0,3/20.,1/4.,1/4.,1/4.,1/10.,0])


def get_PREF_tensor_pizzaNsalad():
    labels = ['pizza','indian','salad','sushi']
    
    PREF_tensor = np.zeros((4,4,2))
    
    PREF_tensor[:,:,0]= np.array([[0.5, 0.5, 0.99, 0.69], 
                           [0.5, 0.5, 0.59, 0.79], 
                           [0.01, 0.41, 0.5, 0.2], 
                           [0.31, 0.21, 0.8, 0.5]])
    
    PREF_tensor[:,:,1]= np.array([[0.5, 0.5, 0.99, 0.69], 
                           [0.5, 0.5, 0.59, 0.79], 
                           [0.01, 0.41, 0.5, 0.6], 
                           [0.31, 0.21, 0.4, 0.5]])
    
    return PREF_tensor, labels

def get_PREF_tensor_football1():
    labels = ['Tunisia','Malta','Brazil','Argentina']
    
    PREF_tensor = np.zeros((4,4,2))
    
    PREF_tensor[:,:,0]= np.array([[0.5, 0.6, 0.3, 0.1], 
                           [0.4, 0.5, 0.25, 0.05], 
                           [0.7, 0.75, 0.5, 0.55], 
                           [0.9, 0.95, 0.45, 0.5]])
    
    PREF_tensor[:,:,1]= np.array([[0.5, 0.55, 0.25, 0.05], 
                           [0.45, 0.5, 0.25, 0.05], 
                           [0.75, 0.75, 0.5, 0.58], 
                           [0.95, 0.95, 0.42, 0.5]])
    
    return PREF_tensor, labels

def get_PREF_tensor_football2():
    labels = ['Tunisia','Malta','Brazil','Argentina']
    
    PREF_tensor = np.zeros((4,4,2))
    
    PREF_tensor[:,:,0]= np.array([[0.5, 0.6, 0.2, 0.1], 
                           [0.4, 0.5, 0.15, 0.05], 
                           [0.8, 0.85, 0.5, 0.55], 
                           [0.9, 0.95, 0.45, 0.5]])
    
    PREF_tensor[:,:,1]= np.array([[0.5, 0.55, 0.15, 0.05], 
                           [0.45, 0.5, 0.2, 0.05], 
                           [0.85, 0.8, 0.5, 0.58], 
                           [0.95, 0.95, 0.42, 0.5]])
    
    return PREF_tensor, labels

def get_PREF_tensor_footballP(eta):
    labels = ['Tunisia','Malta','Brazil','Argentina']
    
    PREF_tensor = np.zeros((4,4,2))
    
    PREF_tensor[:,:,0]= np.array([[0.5, 0.6, 0.3-eta, 0.1], 
                           [0.4, 0.5, 0.25-eta, 0.05], 
                           [0.7+eta, 0.75+eta, 0.5, 0.55], 
                           [0.9, 0.95, 0.45, 0.5]])
    
    PREF_tensor[:,:,1]= np.array([[0.5, 0.55, 0.25-eta, 0.05], 
                           [0.45, 0.5, 0.25-eta, 0.05], 
                           [0.75+eta, 0.75+eta, 0.5, 0.58], 
                           [0.95, 0.95, 0.42, 0.5]])
    
    return PREF_tensor, labels
