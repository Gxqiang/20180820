# -*- coding: utf-8 -*

import pandas as pd 
import numpy as np 
import random
from SupervisedLaplacianScore import *
import sklearn 


def count_u(l):
	l = list(l)
	l.sort()
	n = len(l)
	re = n*(n-1.)/2.0
	if n == 1:
		return re 
	
	for i in list(set(l)):
		m = l.count(i)
		re -= m*(m-1.)/2.0
	return re 



def coef_unalikeability(x):
	m,n = x.shape 
	u1 = 0.
	for i in range(n):
		r = x[:,i]
		u1 += count_u(r)
	u1 = u1/(m**2)
	return u1*2

def weighted_ft(l_f):
	m = len(l_f)
	re = []
	for i in range(m):
		re.append([l_f[i],2.0*(m-i)/(m*(m+1.0))])

	return re


def EFSA(x, y, threshold=0.5, alpha = 0.5):
	d = {}
	n, m = x.shape 
	final_score = np.zeros(m) # wieghts of features
	for i in list(set(y)):
		d[i] = []
	for j in range(n):
		d[y[j]].append(j)
	num_M1 = int(round(coef_unalikeability(x)*6.5)) #(待补充num_M1>0.05时的adjustment程序)
	if num_M1>0.05*n:
		num_M1 = int(round((num_M1*n)/(num_M1+n+1.0)))
	num_mi = {}
	s1 = {}
	l_sample = []
	# 每个决策类应抽取的样本数
	for k in d.keys():
		random.shuffle(d[k])
		num_mi[k] = int(round(len(d[k])*num_M1/n))
		s1[k] = d[k][:num_mi[k]]
		d[k] = d[k][num_mi[k]:]
		# print len(s1[k])
		n -= num_mi[k] 
		l_sample.extend(s1[k])
	l_feature = LaplacianScore(x[l_sample],y[l_sample], threshold)
	print l_feature
	for i,w in weighted_ft(l_feature):
		final_score[i] += w 
		
	sj = {}
	while n>=num_M1:# j>+2
		l_sample = []
		for k in d.keys():
			random.shuffle(s1[k])
			alpha_m = int(round(num_mi[k]*alpha))
			sj[k] = s1[k][:alpha_m]
			random.shuffle(d[k])
			sj[k].extend(d[k][:(num_mi[k]-alpha_m)])
			d[k] = d[k][(num_mi[k]-alpha_m):]
			n -= (num_mi[k]-alpha_m)
			s1[k] = sj[k]
			# print len(s1[k])
			l_sample.extend(s1[k])
		l_feature = LaplacianScore(x[l_sample],y[l_sample],threshold)
		print l_feature
		for i,w in weighted_ft(l_feature):
			final_score[i] += w 
	
	return final_score

