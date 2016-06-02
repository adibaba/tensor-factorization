import numpy as np
from scipy import linalg

from config import num_k, lr, num_R_core

def RUpdate(X, A, R):
    # computes the updates of the Core tensor, uncomment for regularized factorization
    Q, A_hat = linalg.qr(A, overwrite_a=True, mode='economic')
    #Z = np.kron(A_hat,A_hat)
    Z = np.dot(A_hat.T,A_hat)
    for i in range(num_k): 
        ##vec = 0.5*np.dot(Z.T, np.dot(Q.T,(X[i]+Y[i]).dot(Q)).ravel())
        #R[i] = linalg.solve(np.dot(Z.T,Z) + lr*np.eye(num_R_core**2, num_R_core**2), vec).reshape((num_R_core, num_R_core))
        X_hat = np.dot(Q.T, X[i].dot(Q))
        M = linalg.solve(Z, np.dot(np.dot(A_hat.T, X_hat), A_hat), overwrite_b=True, check_finite=False)
        R[i] = linalg.solve(Z, M.T, overwrite_b=True, check_finite=False).T        
    return R

def AUpdate(XY, A, R):
    # computes the updates of the latent embedding
    M = np.sum([XY[i].dot(np.dot(A, R[i].T)) + XY[i].T.dot(np.dot(A, R[i])) for i in range(num_k)], axis=0)
    N = np.sum([np.dot(R[i], np.dot(A.T, np.dot(A,R[i].T))) + np.dot(R[i].T, np.dot(A.T, np.dot(A,R[i]))) for i in range(num_k)], axis=0)
    return linalg.solve((N + lr*np.eye(num_R_core, num_R_core).T), M.T, overwrite_b=True, check_finite=False).T
    #return np.dot(M ,linalg.inv(N + lr*np.eye(num_R_core,num_R_core)))
