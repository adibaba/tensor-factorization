import numpy as np
from scipy import linalg

class RescalFactorizer:
	def __init__(self, num_R_core, lr, min_steps, max_steps, treshold):
		self.num_R_core = num_R_core
		self.lr = lr
		self.min_steps = min_steps
		self.max_steps = max_steps
		self.treshold = treshold
		
	def RUpdate(self, X, A, R):
		# computes the updates of the Core tensor, uncomment for regularized factorization
		Q, A_hat = linalg.qr(A, overwrite_a=True, mode='economic')
		#Z = np.kron(A_hat,A_hat)
		Z = np.dot(A_hat.T,A_hat)
		for i in range(X.shape[0]): 
			##vec = 0.5*np.dot(Z.T, np.dot(Q.T,(X[i]+Y[i]).dot(Q)).ravel())
			#R[i] = linalg.solve(np.dot(Z.T,Z) + lr*np.eye(num_R_core**2, num_R_core**2), vec).reshape((num_R_core, num_R_core))
			X_hat = np.dot(Q.T, X[i].dot(Q))
			M = linalg.solve(Z, np.dot(np.dot(A_hat.T, X_hat), A_hat), overwrite_b=True, check_finite=False)
			R[i] = linalg.solve(Z, M.T, overwrite_b=True, check_finite=False).T        
		return R
		
	def AUpdate(self, X, A, R):
		# computes the updates of the latent embedding
		M = np.sum([X[i].dot(np.dot(A, R[i].T)) + X[i].T.dot(np.dot(A, R[i])) for i in range(X.shape[0])], axis=0)
		N = np.sum([np.dot(R[i], np.dot(A.T, np.dot(A,R[i].T))) + np.dot(R[i].T, np.dot(A.T, np.dot(A,R[i]))) for i in range(X.shape[0])], axis=0)
		return linalg.solve((N + self.lr*np.eye(self.num_R_core, self.num_R_core).T), M.T, overwrite_b=True, check_finite=False).T
		#return np.dot(M ,linalg.inv(N + lr*np.eye(num_R_core,num_R_core)))

	def factorize(self, X):
		num_k = X.shape[0]
		num_R = X[0].shape[0]
		A = np.random.rand(num_R, self.num_R_core)/(self.num_R_core*num_R)
		R = np.random.randn(num_k,self.num_R_core,self.num_R_core)
		error = 1
		i = 0 
		L = [0]*10
		while i < self.max_steps:
			if i < self.min_steps:
				i+=1            
				error = np.linalg.norm(A - self.AUpdate(X,A,R)) + np.linalg.norm(R - self.RUpdate(X,A,R)) 
				A = self.AUpdate(X,A,R)
				R = self.RUpdate(X,A,R)
				L[i%10] = error
				print("Error: ", error)
				print("relative Improvement on AverageError: ", np.abs(np.mean(np.array(L)) - error)/error)
			elif np.abs(np.mean(np.array(L)) - error)/error > self.treshold:
				i+=1            
				error = np.linalg.norm(A - self.AUpdate(X,A,R)) + np.linalg.norm(R - self.RUpdate(X,A,R)) 
				A = self.AUpdate(X,A,R)
				R = self.RUpdate(X,A,R)
				L[i%10] = error
				print("Error: ", error)
				print("relative Improvement on AverageError: ", np.abs(np.mean(np.array(L)) - error)/error)
			else: break
		return (A,R)

	

		
