from scipy import io
from scipy import sparse
import numpy as np
import pickle as pkl
from config import data_dir
import math
import os
from copy import deepcopy

def load_tensordata(data, ent, pro):
	return np.array([io.mmread(data_dir + data + "/" + str(rel_slice) + ".mtx") for rel_slice in range(1, -1 + len(os.listdir(data_dir + data)))]), pkl.load(open(data_dir + data + "/" + ent + ".pkl", "rb")), pkl.load(open(data_dir + data + "/"+ pro + ".pkl", "rb"))

	
def create_fold(n_folds, n_fold, X, perm, rel, ent):
	K = X[rel].tolil().T
	M = deepcopy(K)
	#the n-th batch of the permutation
	r = math.floor(len(perm)/n_folds)	
	if n_fold == n_folds:
		nperm = perm[r*n_fold:]
	else:
		nperm = perm[r*(n_fold -1):r*n_fold]

	
	for index in perm:
		if index in nperm:
			K.data[ent][index] = 0
		else:
			M.data[ent][index] = 0

	K = K.tocsr().T
	M = M.tocsr().T

	print(np.sum(M.tocsr().getcol(ent).T.todense()))
	print(np.sum(K.tocsr().getcol(ent).T.todense()))
	
	test = np.array([X[i] if i != rel else M for i in range(len(X)) ])
	train = np.array([X[i] if i != rel else K for i in range(len(X))])
	return test, train, nperm
	
	
