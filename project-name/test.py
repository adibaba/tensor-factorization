from utils import load_permutation, load_tensordata, create_fold
from evaluation.evaluation import fmeasure
from factorizations.rescal import RescalFactorizer
import numpy as np
import matplotlib.pyplot as plt

T, rels, ents = load_tensordata("umls")
perm = load_permutation("umls")
#perform 10 fold cross-validation
"""
for n in range(1,11):
	train = create_fold(10, n, T, perm)
	rf = RescalFactorizer(30, 0.01, 1, 3,1e-04)
	A,C = rf.factorize(train)
	print("fm, prec, recall : ", fmeasure(T,A,C, 0.4))
"""
#create AUCPR
train = create_fold(10, 1, T, perm)
rf = RescalFactorizer(100, 0.01, 1, 10,1e-04)
A,C = rf.factorize(train)
x= []
y= []
for tresh in np.linspace(0.1,1,20):
	(f,p,r) = fmeasure(T,A,C,tresh)
	x.append(r)
	y.append(p)

AUCPR = np.trapz(x,y)
plt.plot(x,y)
plt.ylabel('precision')
plt.xlabel('recall, AOCPR = ' + str(AUCPR))
plt.show()	
