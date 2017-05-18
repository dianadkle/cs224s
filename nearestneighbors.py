from sklearn.neighbors import KNeighborsClassifier
from numpy import random,argsort,sqrt
import numpy as np
from pylab import plot,show
import pickle
import sys

def verifyAccuracy(labels, actual):
	assert(len(labels)==len(actual))
	a = 0.0
	c = 0.0
	f = 0.0
	j = 0.0
	s = 0.0
	for i in xrange(len(labels)):
		if labels[i] == actual[i]:
			if labels[i] == 0:
				a += 1
			elif labels[i] == 1:
				c += 1
			elif labels[i] == 2:
				f += 1
			elif labels[i] == 3:
				j += 1
			else:
				s += 1
	print "Arabic: " + str(a/20)
	print "Cantonese: " + str(c/20)
	print "French: " + str(f/20)
	print "Japanese: " + str(j/20)
	print "Spanish: " + str(s/20)
	print "Total correct: " + str((a+c+f+j+s)/100)


def fillXY(data, label):
	i = 0
	for sample, features in data.items():
		if i == 100: break
		if i <80:
			x.append(features)
			if label == 'AR':
				y.append(0)
			elif label == 'CA':
				y.append(1)
			elif label == 'FR':
				y.append(2)
			elif label == 'JA':
				y.append(3)
			else:
				y.append(4)
		else:
			test.append(features)
			if label == 'AR':
				actual.append(0)
			elif label == 'CA':
				actual.append(1)
			elif label == 'FR':
				actual.append(2)
			elif label == 'JA':
				actual.append(3)
			else:
				actual.append(4)
		i+=1

def knn_search(x, D, K):
	""" find K nearest neighbours of data among D """
	# print D
	mindist = sys.maxint
	nearest = 0
	for i in xrange(len(D)):
		sq = (x-D[i])**2
		# print sq
		dist = sqrt(sum(sum(x) for x in sq))
		# print dist
		if dist < mindist:
			mindist = dist
			nearest = y[i]
	# return the indexes of K nearest neighbours
	return nearest

ar = pickle.load(open('data/mfcc_features/AR', 'rb'))
ca = pickle.load(open('data/mfcc_features/CA', 'rb'))
fr = pickle.load(open('data/mfcc_features/FR', 'rb'))
ja = pickle.load(open('data/mfcc_features/JA', 'rb'))
sp = pickle.load(open('data/mfcc_features/SP', 'rb')) 

print "Loaded all data"

x = [] # training data
y = [] # labels
test = []
actual = []
pred = []

fillXY(ar, 'AR')
fillXY(ca, 'CA')
fillXY(fr, 'FR')
fillXY(ja, 'JA')
fillXY(sp, 'SP')

print "Filled the training, labels, pred, and actual"

# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(x, y) 
print len(test)
for i in xrange(len(test)):
	p = knn_search(test[i], x, 1)
	pred.append(p)
print pred
verifyAccuracy(pred, actual)

