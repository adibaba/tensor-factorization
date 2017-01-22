import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import svds

max_steps = 2
min_steps = 1
tresh = 1e-04
lmbdaA = 0.3

def RUpdate(X, A, R):
	# computes the updates of the Core tensor
	def unregularizedRUpdate(A,B):
		#Here we solve ASA.T = X, using that A is upper triangular
		M1 = linalg.solve_triangular(A_hat, X_hat, overwrite_b=True, check_finite=False)
		return linalg.solve_triangular(A_hat, M1.T, overwrite_b=True, check_finite=False).T
	Q, A_hat = np.linalg.qr(A, mode='reduced')
	for i in range(X.shape[0]): 
		X_hat = np.dot(Q.T, X[i].dot(Q))
		R[i] = unregularizedRUpdate(A_hat, X_hat)
	return R
		
def AUpdate(X, A, R):
	# computes the updates of the latent embedding
	AtA = np.dot(A.T, A)
	n, rank = A.shape
	M = np.zeros((n, rank), dtype=A.dtype)
	N = np.zeros((rank, rank), dtype=A.dtype)
	for i in range(X.shape[0]):
		M += X[i].dot(np.dot(A, R[i].T)) + X[i].T.dot(np.dot(A, R[i]))
		N += np.dot(R[i], np.dot(AtA,R[i].T)) + np.dot(R[i].T, np.dot(AtA,R[i]))
	return linalg.solve((N + lmbdaA*np.eye(rank, rank)), M.T, overwrite_b=True, check_finite=False).T

def error(X, A, R, normX):
	error= 0 
	AtA = np.dot(A.T,A)
	for i in range(X.shape[0]):
		AtARt = np.dot(AtA,R[i].T)
		AtAR  = np.dot(AtA,R[i])
		AtXtA = np.dot(A.T, X[i].T*A)
		error += np.trace(np.dot(AtARt,AtAR)) - 2*np.trace(np.dot(AtXtA, R[i]))
	return 1 + error/normX

def factorize(X, d):
	num_k = X.shape[0]
	num_R = X[0].shape[0]
	normX = np.sum([X[i].getnnz() for i in range(num_k)])**2
	print("initializing R and A")
	R = np.random.randn(num_k, d, d)
	A= np.random.randn(num_R, d)
	eprev = 0
	for i in range(max_steps+1):
		print("Iteration : ", i)
		print("... AUpdate...")
		A = AUpdate(X, A, R)
		print("... RUpdate...")
		R = RUpdate(X, A, R)
		print("... Error...")
		e = error(X, A, R, normX)
		print("Error : ", e)
		#print("Improvement : ", np.abs(eprev-e))
		if i > min_steps:
			if np.abs(eprev-e) < tresh:
				break
		eprev = e
	return (A,R)

