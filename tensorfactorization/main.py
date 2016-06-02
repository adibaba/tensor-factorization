import os
import numpy as np
import scipy
from scipy import sparse as sparse
from scipy import linalg
from scipy import io

from tensorupdates import AUpdate, RUpdate
from config import lr, num_R, num_k, num_R_core, threshold, max_steps, min_steps, rounding_threshold
from config import data_dir

def factorize_tensor():
    X = np.array([io.mmread(os.path.join(data_dir,'A70%sider_dump' + str(k) + '.mtx')) for k in range(num_k)])
    Y = np.array([io.mmread(os.path.join(data_dir,'B70%sider_dump' + str(k) + '.mtx')) for k in range(num_k)])
    FULL = np.array([io.mmread(os.path.join(data_dir,'Full_sider_dump' + str(k) + '.mtx')) for k in range(num_k)])
    XY = np.array([(0.5*(X[i]+Y[i])).ceil() for i in range(num_k)])

    A = np.random.rand(num_R, num_R_core)/(num_R_core*num_R)
    R = np.random.randn(num_k,num_R_core,num_R_core)

    error = 1
    i = 0 
    L = [0]*10
    while i < max_steps:
            if i < min_steps:
                    i+=1            
                    error = np.linalg.norm(A - AUpdate(XY,A,R)) + np.linalg.norm(R - RUpdate(XY,A,R)) 
                    A = AUpdate(XY,A,R)
                    R = RUpdate(XY,A,R)
                    L[i%10] = error
                    print("Error: ", error)
                    print("relative Improvement on AverageError: ", np.abs(np.mean(np.array(L)) - error)/error)
            elif np.abs(np.mean(np.array(L)) - error)/error > threshold:
                    i+=1            
                    error = np.linalg.norm(A - AUpdate(XY,A,R)) + np.linalg.norm(R - RUpdate(XY,A,R)) 
                    A = AUpdate(XY,A,R)
                    R = RUpdate(XY,A,R)
                    L[i%10] = error
                    print("Error: ", error)
                    print("relative Improvement on AverageError: ", np.abs(np.mean(np.array(L)) - error)/error)
            else: break
    return (FULL, A, R)

factorize_tensor()
