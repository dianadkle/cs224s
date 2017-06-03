from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

min_seq_length = 200  ## can be changed to 50 if 50 is used in mfcc_extraction.py
dirData = "data-200"  ## 'data-50' the directory where mfcc_features, delta_features, fbank_features reside
learning_rate_init_tuning = True


print 'min_seq_length=', min_seq_length

def fillXY(mfcc, delta, fbank, label):
	i = 0
	dev_count = 0
	test_count = 0
	train_count = 0
	# note that for TA, GE, HI, BP, SP set, we have at least 308 samples in each language.
    # you can change the index to record the effect of using less data
    # the code below skips the sample with length less than min_seq_length
	for j in xrange(len(mfcc)):
		if len(mfcc[j]) != min_seq_length:
		    print "bad mfcc: y =", label, "sample=", j, "shape =", len(mfcc[j])
		    continue
           
            
		if len(delta[j]) != min_seq_length:
		    print "bad delta: y =", label, "sample=", j, "shape =", len(delta[j])
		    continue
           
            
		if len(fbank[j]) != min_seq_length:
		    print "bad fbank: y =", label, "sample=", j, "shape =", len(fbank[j])
		    continue    
                
		if i == 300: break
 		elif i >= 250: # real test set
			test_count = test_count + 1
            #todo can change the following line to only use a subset of features or change the order
            #todo 
            # try features = mfcc[j]
            # features = delta[j]
            # features = fbank[j]
            # then 2 feature sets
			features = np.concatenate((mfcc[j], delta[j], fbank[j]), axis=1)
			realtest.append(features.flatten())
			if label == 'TA':
			    realactual.append(0)
			elif label == 'GE':
			    realactual.append(1)
			elif label == 'BP':
			    realactual.append(2)
			elif label == 'HI':
			    realactual.append(3)
			else:
			    realactual.append(4)
                
		elif i > 20 and i < 71:
			dev_count = dev_count + 1
            #todo can change the following line to only use a subset of features or change the order
            #todo 
            # try features = mfcc[j]
            # features = delta[j]
            # features = fbank[j]
            # then 2 feature sets
			features = np.concatenate((mfcc[j], delta[j], fbank[j]), axis=1)
			test.append(features.flatten())
			if label == 'TA':
				actual.append(0)
			elif label == 'GE':
				actual.append(1)
			elif label == 'BP':
				actual.append(2)
			elif label == 'HI':
				actual.append(3)
			else:
				actual.append(4)
		else:
			#features = np.append(mfcc[j], fbank[j])
			# train
			train_count = train_count + 1
			#todo can change the following line to only use a subset of features or change the order
            #todo 
            # try features = mfcc[j]
            # features = delta[j]
            # features = fbank[j]
            # then 2 feature sets
			features = np.concatenate((mfcc[j], delta[j], fbank[j]), axis=1)
			x.append(features.flatten())
			if label == 'TA':
				y.append(0)
			elif label == 'GE':
				y.append(1)
			elif label == 'BP':
				y.append(2)
			elif label == 'HI':
				y.append(3)
			else:
				y.append(4)
		i+=1
 	print "train_count for ", label, "=", train_count, "dev_count=", dev_count, "test_count=", test_count   
	return x, y, test, label



ar = pickle.load(open(dirData + '/mfcc_features/TA', 'rb'))
ar_delta = pickle.load(open(dirData + '/delta_features/TA', 'rb'))
ar_fbank = pickle.load(open(dirData + '/fbank_features/TA', 'rb'))

ca = pickle.load(open(dirData + '/mfcc_features/GE', 'rb'))
ca_delta = pickle.load(open(dirData + '/delta_features/GE', 'rb'))
ca_fbank = pickle.load(open(dirData + '/fbank_features/GE', 'rb'))

fr = pickle.load(open(dirData + '/mfcc_features/BP', 'rb'))
fr_delta = pickle.load(open(dirData + '/delta_features/BP', 'rb'))
fr_fbank = pickle.load(open(dirData + '/fbank_features/BP', 'rb'))

ja = pickle.load(open(dirData + '/mfcc_features/HI', 'rb'))
ja_delta = pickle.load(open(dirData + '/delta_features/HI', 'rb'))
ja_fbank = pickle.load(open(dirData + '/fbank_features/HI', 'rb'))

sp = pickle.load(open(dirData + '/mfcc_features/SP', 'rb')) 
sp_delta = pickle.load(open(dirData + '/delta_features/SP', 'rb'))
sp_fbank = pickle.load(open(dirData + '/fbank_features/SP', 'rb'))

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

# unseen test set
realtest = []
realactual = []

fillXY(ar, ar_delta, ar_fbank, 'TA')
fillXY(ca, ca_delta, ca_fbank, 'GE')
fillXY(fr, fr_delta, fr_fbank,  'BP')
fillXY(ja, ja_delta, ja_fbank, 'HI')
fillXY(sp, sp_delta, sp_fbank, 'SP')



#clf = MLPClassifier(hidden_layer_sizes=(100,), activation='identity', solver='sgd',learning_rate='adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(100),  solver='sgd', random_state=101) 0.26
#clf = MLPClassifier(hidden_layer_sizes=(256, 256, 256, 256, 256, 100),  solver='adam', random_state=102, learning_rate_init = 0.0001, max_iter = 1000, early_stopping=False, validation_fraction=0 )# score 0.38

#### change below
hidden_layer_sizes = (100, 100, 100)
random_state=103
learning_rate_init = 0.0001
max_iter = 1000
early_stopping=False
validation_fraction=0
solver = 'adam' # 'sgd'
alpha = 0.0001
#### end of paramters

print "y=", len(y),"actual=", len(actual), "realactual=", len(realactual)

print 'hidden_layer_sizes =', hidden_layer_sizes
print 'random_state=', random_state
print 'learning_rate_init=', learning_rate_init
print 'max_iter=', max_iter
print 'early_stopping=', early_stopping
print 'validation_fraction=', validation_fraction
print 'solver=', solver
print 'alpha =', alpha

clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,  solver=solver, random_state=random_state, learning_rate_init = learning_rate_init, max_iter = max_iter, early_stopping=early_stopping, validation_fraction=validation_fraction, alpha = alpha ) 
clf.fit(x,y)  

print "train  ",  "SCORE IS: " + str(clf.score(x, y))
print "test ",  "SCORE IS: " + str(clf.score(test, actual))
print "realtest ", "SCORE IS: " + str(clf.score(realtest, realactual))

if learning_rate_init_tuning :
    print "---Hyperparameter training  (use training  to train and get test accuracy "
    for hidden_layer_sizes in [(100), (100, 100), (100, 100, 100), (256), (256, 256), (256, 256, 256), (256, 256, 256, 100)]:  #todo please add more
        for learning_rate_init in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
            for alpha in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
                print "hidden layer = ", hidden_layer_sizes
                print "learning_rate_init = ", learning_rate_init
                print "alpha = ", alpha
                clf1 = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,  solver=solver, random_state=random_state, learning_rate_init = learning_rate_init, max_iter = max_iter, early_stopping=early_stopping, validation_fraction=validation_fraction, alpha = alpha ) 
                clf1.fit(x,y)  

                print "test ",  "SCORE IS: " + str(clf1.score(test, actual))
                print "----------"
   

### please add other hyperparameter training 
# max_iter [1000] 1000 seems to be working
# alpha [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

### realtest and realactual will be used to get the score once hyperparameters are tuned.