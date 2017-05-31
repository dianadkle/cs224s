from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

def fillXY(mfcc, delta, label):
	i = 0
	# for sample, features in mfcc.items():
	for j in xrange(len(mfcc)):
		if i == 100: break
		if i > 20 and i < 41:
			features = np.append(mfcc[j], delta[j])
			test.append(features.flatten())
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
		else:
			features = np.append(mfcc[j], delta[j])
			x.append(features.flatten())
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
		i+=1
	return x, y, test, label


ar = pickle.load(open('data/mfcc_features/AR', 'rb'))
ar_delta = pickle.load(open('data/delta_features/AR', 'rb'))
ca = pickle.load(open('data/mfcc_features/CA', 'rb'))
ca_delta = pickle.load(open('data/delta_features/CA', 'rb'))
fr = pickle.load(open('data/mfcc_features/FR', 'rb'))
fr_delta = pickle.load(open('data/delta_features/FR', 'rb'))
ja = pickle.load(open('data/mfcc_features/JA', 'rb'))
ja_delta = pickle.load(open('data/delta_features/JA', 'rb'))
sp = pickle.load(open('data/mfcc_features/SP', 'rb')) 
sp_delta = pickle.load(open('data/delta_features/SP', 'rb'))

print "Loaded all data"

# arx, ary, artest, arlabel = fillXY(ar, 'AR')
# cax, cay, catest, calabel = fillXY(ca, 'CA')
# frx, fry, frtest, frlabel = fillXY(fr, 'FR')
# jax, jay, jatest, jalabel = fillXY(ja, 'JA')
# spx, spy, sptest, splabel = fillXY(sp, 'SP')

x = [] # training data
y = [] # labels
test = []
actual = []
pred = []

fillXY(ar, ar_delta, 'AR')
fillXY(ca, ca_delta, 'CA')
fillXY(fr, fr_delta, 'FR')
fillXY(ja, ja_delta, 'JA')
fillXY(sp, sp_delta, 'SP')



clf = MLPClassifier(hidden_layer_sizes=(100,), activation='identity', solver='sgd',learning_rate='adaptive')

clf.fit(x,y)

print "SCORE IS: " + str(clf.score(test, actual))

