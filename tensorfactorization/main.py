import numpy as np
import scipy
from scipy import io
from scipy import sparse as sparse
from scipy import linalg

lr = 0.01
num_R = 30378
num_k = 11
num_R_core = 20
treshhold = 0.07
max_steps = 500
min_steps = 6
rounding_treshhold = 0.3

def RUpdate(X, A, R):
    #
    # computes the updates of the Core tensor, uncomment for regularized factorization
    #
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
    #
    # computes the updates of the latent embedding
    #
        M = np.sum([XY[i].dot(np.dot(A, R[i].T)) + XY[i].T.dot(np.dot(A, R[i])) for i in range(num_k)], axis=0)
        N = np.sum([np.dot(R[i], np.dot(A.T, np.dot(A,R[i].T))) + np.dot(R[i].T, np.dot(A.T, np.dot(A,R[i]))) for i in range(num_k)], axis=0)
        return linalg.solve((N + lr*np.eye(num_R_core, num_R_core).T), M.T, overwrite_b=True, check_finite=False).T
    #return np.dot(M ,linalg.inv(N + lr*np.eye(num_R_core,num_R_core)))

def fmeasure(Full,A,R,rounding_treshhold):
    #
    # computes the fmeasure of A@R@A^t  vs Full
    #
        a = 0
        relevant = 0
        retrieved = 0 
        for l in range(num_k):
                for i,j,k in zip(Full[l].row, Full[l].col, Full[l].data):
                        if np.around(np.clip(np.dot(A[i,:], np.dot(R[l],A.T[:,j]))-(0.5-rounding_treshhold), 0,1)) == k:
                                a += 1
                relevant += Full[l].getnnz()
    #
    #  To calculate precision we have to stream the entries... found no better way yet
    #
                for s in range(num_R):
                        retrieved += np.sum(np.clip(np.around(np.dot(A, np.dot(R[l],A.T[:,s]))-(0.5-rounding_treshhold)), 0,1))
        prec =  a/float(retrieved)
        reca = a/float(relevant)
        return 2 * prec*reca/float(prec + reca), prec, reca

#TODO: another package -- core loop
X = np.array([io.mmread('../data/70%/A70%sider_dump' + str(k) + '.mtx') for k in range(num_k)])
Y = np.array([io.mmread('../data/70%/B70%sider_dump' + str(k) + '.mtx') for k in range(num_k)])
FULL = np.array([io.mmread('../data/70%/Full_sider_dump' + str(k) + '.mtx') for k in range(num_k)])
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
        elif np.abs(np.mean(np.array(L)) - error)/error > treshhold:
                i+=1            
                error = np.linalg.norm(A - AUpdate(XY,A,R)) + np.linalg.norm(R - RUpdate(XY,A,R)) 
                A = AUpdate(XY,A,R)
                R = RUpdate(XY,A,R)
                L[i%10] = error
                print("Error: ", error)
                print("relative Improvement on AverageError: ", np.abs(np.mean(np.array(L)) - error)/error)
        else: break

#TODO: another package -- evaluation
rounding_treshhold = 0.2
while 1 - rounding_treshhold > 0.2:
        print(1 - rounding_treshhold) 
        print("fmeasure, precision, recall:", fmeasure(FULL, A, R,rounding_treshhold))
        rounding_treshhold += .1

