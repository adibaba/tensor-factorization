import config
import numpy as np
from scipy import linalg
from scipy import sparse

#import utils
from evaluation.evaluation import fmeasure
from factorizations.rescal import RescalFactorizer

X = np.array([sparse.coo_matrix(np.eye(200,200).round().clip(0,1)) for k in range(1)])
rf = RescalFactorizer(20, 0.01, 10, 100,1e-04)
A,C = rf.factorize(X)
Q,R = linalg.qr(A)
U,D,V = linalg.svd(R)

print("condition number of A: ", np.max(D)/np.min(D))

print(np.diagonal(np.dot(A,C[0].dot(A.T))) )
print("fm, prec, recall : ", fmeasure(X,A,C, 0.08))
