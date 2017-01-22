import numpy as np
import theano
import theano.tensor as T


def fmeasure(X,A,R,rounding_threshold):
	# computes the fmeasure of A@R@A^t  vs Full
	a = 0
	relevant = 0
	retrieved = 0 
	for l in range(X.shape[0]):
		for i,j,k in zip(X[l].row, X[l].col, X[l].data):
			if np.around(np.clip(np.dot(A[i,:], np.dot(R[l],A.T[:,j]))+(0.5-rounding_threshold), 0,1)) == k:
				a += 1
		relevant += X[l].getnnz()
		#  To calculate precision we have to stream the entries... found no better way yet		
		for s in range(X[0].shape[0]):
			retrieved += np.sum(np.clip(np.around(np.dot(A, np.dot(R[l],A.T[:,s]))+(0.5-rounding_threshold)), 0,1))

	prec =  a/float(retrieved)
	reca = a/float(relevant)
	return 2 * prec*reca/float(prec + reca), prec, reca

"""
def retrieved(A,R,rounding_treshold):
	A = T.matrix("A")
	R = T.matrix("R")
	k = T.iscalar("k")
	result, update = theano.scan(fn= lambda, prior_result, k: np.sum((A.dot(R).dot(A.T[:,k])+(0.5-rounding_threshold)).round().clip(0,1)) + prior_result, 
					n_steps = k)
	final_result = result[-1]
	retrieved = theano.function(inputs=[A,k], outputs=final_result)
	return retrieved
"""
