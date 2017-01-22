import numpy as np


def fmeasure(X,A,R,rounding_threshold, l):
	triples = set(zip(X[l].row, X[l].col, X[l].data))
	relevant = len(triples)	
	retrieved = 0
	a = 0 
	for i,j,k in triples:
		if np.around(np.clip(np.dot(A[i,:], np.dot(R[l],A.T[:,j]))+(0.5-rounding_threshold), 0,1)) == k:
			a += 1
	#get the vectors for subjects (row indices) and objects ( col indices ) 
	Ar = np.array([A[i] for i in set(X[l].row)])
	Ac = np.array([A[j] for j in set(X[l].col)])
	RAct = np.dot(R[l], Ac.T)
	for i in range(len(Ar)):
		resourcerating = np.dot(Ar[i], RAct)
		reconstructed = np.clip(np.around(resourcerating+(0.5-rounding_threshold)), 0,1)
		retrieved += np.sum(reconstructed*reconstructed)
	reca = a/float(relevant)
	try:	
		prec =  a/float(retrieved)
	except ZeroDivisionError:
		prec = 0	
	try:	
		fm =  2 * prec*reca/float(prec + reca)
	except ZeroDivisionError:
		fm = 0
	
	return fm, prec, reca

def AUCPR(T,A,R):
	x= []
	y= []
	for tresh in np.linspace(0,1,200):
		(f,p,r) = fmeasure(T,A,R,tresh)
		y.insert(0,p)
		x.insert(0,r)
	x.append(1)
	y.append(0)
	x.insert(0,0)
	y.insert(0,y[0])
	AUCPR = np.trapz(y,x)
	return AUCPR

def axelPR(X, test, A, R, relation, entity, fold, nperm):
	xVector = np.array(X[relation].tocsr().getcol(entity).T.todense())[0]
	testVector = np.array(test[relation].tocsr().getcol(entity).T.todense())[0]
	entityVector = np.dot(np.dot(A, R[relation]), A.T[:,entity])
	sortedVector = sorted(list(enumerate(entityVector)), key = lambda x: -x[1])
	print(sortedVector[0:10])
	TP = 0
	FP = 0
	AllFacts = np.sum(testVector)
	print(AllFacts)
	Ps = []
	Rs = []
	FPs = []
	TPs = []
	i=1
	for index, element in (sortedVector):
		if  testVector[index] == 1:
			TP += 1
			TPs.append(tuple([i, index, element]))
			i+=1
		else:
			if not xVector[index] == 1:
				FP += 1
				FPs.append(tuple([i, index, element]))
				i+=1
		if TP == 0:
			P = 0
		else:
			P = TP/float(TP + FP)
		R = TP/float(AllFacts)
		Ps.append(P)
		Rs.append(R)
	return Ps, Rs, FPs, TPs
	
