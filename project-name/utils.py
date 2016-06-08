from scipy import io
from scipy import sparse
import numpy as np
import pickle as pkl
from config import data_dir
import math

def load_tensordata(data):
	with open( data_dir + data + "/" +data +".pkl", "rb") as f:
		Tdict = pkl.load(f, encoding="latin1")
	f.close()
	T = Tdict["tensor"]
	T = np.array([sparse.coo_matrix(T[:,:,k]) for k in range(T.shape[2])])
	try:	
		rels = Tdict["attname"]
		ents = Tdict["relname"]
	except KeyError as e:
		print("Run-Time Error: ", e)
		rels = []
		ents = []
	return T,rels, ents

def load_permutation(data):
	try:
		with open( data_dir + data + "/" +data +"_permutations.pkl", "rb") as f:
			perms = pkl.load(f, encoding="latin1")
		f.close()
	except FileNotFoundException as e:
		print("Run-Time Erorr: ", e)
		perms = []
	return perms


#def tensor_from_graph():
	
def create_fold(n_folds, n_fold, T, perm):
	# creates train, test splits for n-fold cross-validation 
	# percentage % Test, rest Train, "n_fold" starts from 1
	
	#create train and test, csr allows for train[k][i,j] = x assignement
	train = np.array([T[k].tocsr() for k in range(T.shape[0])])	
	#the n-th batch of the permutation
	r = math.floor(len(perm)/n_folds)	
	if n_folds == n_fold:
		nperm = perm[r*n_fold:]
	else:
		nperm = perm[r*(n_fold -1):r*n_fold]

	#a helper list to identify the link that is to be removed	
	indhelp = []
	for k in range(T.shape[0]):  
		indhelp.append(len(T[k].data))

	#remove the nperm[k]-th entry out out of the 'flattened' tensor
	for k in nperm:
		for n in range(len(indhelp)): 
			if k > indhelp[n]:
				k += -indhelp[n]
			else:
				i,j = (T[n].row[k-1],T[n].col[k-1])
				train[n][i,j] = 0
				break
	#convert to coo matrix before returning
	train = np.array([train[k].tocoo() for k in range(T.shape[0])])	
	return train
	
	
