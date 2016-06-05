import numpy as np

from main import factorize_tensor
from config import num_k, rounding_threshold

def fmeasure(Full,A,R,rounding_threshold):
    # computes the fmeasure of A@R@A^t  vs Full
    a = 0
    relevant = 0
    retrieved = 0 
    for l in range(num_k):
        for i,j,k in zip(Full[l].row, Full[l].col, Full[l].data):
            if np.around(np.clip(np.dot(A[i,:], np.dot(R[l],A.T[:,j]))-(0.5-rounding_threshold), 0,1)) == k:
                a += 1
        relevant += Full[l].getnnz()
        #  To calculate precision we have to stream the entries... found no better way yet
        for s in range(num_R):
            retrieved += np.sum(np.clip(np.around(np.dot(A, np.dot(R[l],A.T[:,s]))-(0.5-rounding_threshold)), 0,1))

    prec =  a/float(retrieved)
    reca = a/float(relevant)
    return 2 * prec*reca/float(prec + reca), prec, reca

(FULL, A, R) = factorize_tensor()
rounding_threshold = 0.2
while 1 - rounding_threshold > 0.2:
        print(1 - rounding_threshold) 
        print("fmeasure, precision, recall:", fmeasure(FULL, A, R, rounding_threshold))
        rounding_threshold += .1

