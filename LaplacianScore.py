# -*- coding: utf-8 -*
import numpy as np 
import pandas as pd 

def LaplacianScore(x,y):
	# input: x (n_sample*n_feature),(numpy.ndarray), 特征矩阵; y, 类别标签
	# output: 对应的Laplacian Score
	y = list(y)
	n_sample,n_feature = x.shape

	# 计算相似矩阵
	S = np.zeros([n_sample,n_sample])
	l_class = list(set(y))
	d_class = {}
	for i in l_class:
		d_class[i] = y.count(i)
	for i in range(n_sample):
		for j in range(i+1,n_sample):
			if y[i]==y[j]:
				S[i,j] = d_class[y[i]]
	S = S + S.T
	D = np.diag(sum(S))
	L = D - S
	I = np.ones([1,n_sample])
	Ls = np.zeros(n_feature)
	
	
	for r in range(n_feature):
		f_r = np.array([x[:,r]])
		f_r_s = f_r - (f_r.dot(D.dot(I.T))[0,0]/np.sum(D)) * I #标准化
		Ls[r] = (f_r_s.dot(L.dot(f_r_s.T))[0,0])/(f_r_s.dot(D.dot(f_r_s.T))[0,0])
	
	return Ls

data = pd.read_csv("F:\IRIS.csv")
df= data.as_matrix()
ls = LaplacianScore(df[:,:-1],df[:,-1])
print ls

from sklearn.feature_selection import SelectKBest ,chi2
#选择相关性最高的前5个特征
X_chi2 = SelectKBest(chi2, k=1).fit_transform(df[:,:-1],df[:,-1])

print X_chi2