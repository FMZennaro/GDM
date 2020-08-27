
import numpy as np

def normalize_by_column(A):
    # Make the matrix A left stochastic
    # (column summing up to 1)
    return A / np.sum(A,axis=0)

def normalize_by_column_tensor(PREF_tensor):
    # INPUT:
    # PREF_tensor    [MxMxE] tensor (choice, choice, expert)
    #
    # OUTPUT:
    # PROB_tensor    [MxMxE] tensor (choice, choice, expert)
    PROB_tensor = np.zeros(PREF_tensor.shape)
    for i in range(0,PREF_tensor.shape[2]):
        n = normalize_by_column(PREF_tensor[:,:,i])
        PROB_tensor[:,:,i] = n
    return PROB_tensor

def centrality_matrix(A,M):
    # Make the matrix A left stochastic
    # (column summing up to 1)
    #
    # This code has side effects!
    C=A.copy()
    D=A.copy()

    #print "Before transition", A
    sum_row=[0]*M
    
    for ix,iy in np.ndindex(D.shape):
        if ix!=iy:
            D[ix,iy]=(1.0/(M-1))*(1-C[ix,iy])
            
            sum_row[ix]=sum_row[ix]+D[ix,iy]
            #print "sum row",sum_row[ix]
            
    for ix,iy in np.ndindex(D.shape):
         if ix==iy:
            D[ix,iy]=1-sum_row[ix]

    return D

def centrality_tensor(PREF_tensor):
    # INPUT:
    # PREF_tensor    [MxMxE] tensor (choice, choice, expert)
    #
    # OUTPUT:
    # PROB_tensor    [MxMxE] tensor (choice, choice, expert)
    PROB_tensor = np.zeros(PREF_tensor.shape)
    for i in range(0,PREF_tensor.shape[2]):
        n = centrality_matrix(PREF_tensor[:,:,i],PREF_tensor.shape[0])
        PROB_tensor[:,:,i] = n
    return PROB_tensor


