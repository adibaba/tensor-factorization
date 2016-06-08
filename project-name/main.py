from utils import load_permutation, load_tensordata, create_fold
from evaluation.evaluation import fmeasure
from factorizations.rescal import RescalFactorizer
import numpy as np
import matplotlib.pyplot as plt

T, rels, ents = load_tensordata("kinships")
perm = load_permutation("kinships")
auc = []
#perform 10 fold cross-validation
for n in range(1,11):
	train = create_fold(10, n, T, perm)
	rf = RescalFactorizer(20 , 0.01, 10, 50,1e-04)
	A,C = rf.factorize(train)
	x= []
	y= []
	for tresh in np.linspace(0,1,200):
		(f,p,r) = fmeasure(T,A,C,tresh)
		y.insert(0,p)
		x.insert(0,r)
	print(x,y)
	x.append(1)
	y.append(0)
	x.insert(0,0)
	y.insert(0,y[0])
	AUCPR = np.trapz(y,x)
	auc.append(AUCPR)
	plt.plot(x,y)
	plt.ylabel('recall')
	plt.xlabel('precision, AOCPR = ' + str(AUCPR))
plt.show()	
print(np.mean(auc))


