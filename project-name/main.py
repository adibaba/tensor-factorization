from utils import load_tensordata, create_fold
from factorizations import rescal
from rdflib import URIRef
import numpy as np
from evaluation.evaluation import fmeasure, axelPR
import time
import pickle
import os

latent_dimensions = 7
tic = time.time()
print("... loading matrices...")
X, e2i, p2i = load_tensordata("dbpediayago", "ent", "pro")
p = "rdf:type"
i_p = p2i[p]-1
e = "<wordnet_person_100007846>"
i_e = e2i[e] -1
print(i_p, i_e)

filename = "Evaluation"+ "Prop_" +str(i_p)+ "Ent_" + str(i_e)
toc = time.time()
print("... ... took {} seconds".format(toc-tic))
tic = time.time()
perm = np.random.permutation(len(X[i_p].tocsr().getcol(i_e).data))
os.makedirs(filename)
for fold in range(1, 6):
	print("... generating train/test split ...")
	test, train, nperm = create_fold(5, fold, X, perm, i_p, i_e)
	print("... computing factorization ...")
	A, R = rescal.factorize(train, latent_dimensions)
	print("... computing AUCPR ...")
	Ps, Rs, FPs, TPs = axelPR(X, test, A, R, i_p, i_e, fold, nperm)
	pickle.dump(Ps, open(filename + "/PsFold" + str(fold) + ".pkl", "wb+"))
	pickle.dump(Rs, open(filename + "/RsFold" + str(fold) + ".pkl", "wb+"))
	pickle.dump(TPs, open(filename + "/TPsFold" + str(fold) + ".pkl", "wb+"))
	pickle.dump(FPs, open(filename + "/FPsFold" + str(fold) + ".pkl", "wb+"))
