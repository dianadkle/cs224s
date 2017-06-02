import numpy as np
from sklearn.model_selection import train_test_split
from numpy import random,argsort,sqrt
from pylab import plot,show
import pickle
import sys, random

class Config:
    min_seq_length = 50 #50 #200

class Label:
    
    arabic = 0
    cantonese = 1
    french = 2
    japanese = 3
    spanish = 4
    
    '''
    tamil = 0 #TA
    germany = 1  #GE
    brazilianP = 2 #BP
    hindi = 3 #HI
    spanish = 4 #SP
    '''
class FeatureExtractor(object):
    def __init__(self, test_split=0.2, sk_split=False):
        self.test_split = test_split
        self.sk_split = sk_split
        self.seq_length, self.embed_size = None, None # (50, 13)

        print "Loading data"
        ar = pickle.load(open('data/mfcc_features/AR', 'rb'))
        ca = pickle.load(open('data/mfcc_features/CA', 'rb'))
        fr = pickle.load(open('data/mfcc_features/FR', 'rb'))
        ja = pickle.load(open('data/mfcc_features/JA', 'rb'))
        sp = pickle.load(open('data/mfcc_features/SP', 'rb')) 
      
        
        '''
        print '**** MFCC features ****'
        ta = pickle.load(open('data/mfcc_features/TA', 'rb'))
        ge = pickle.load(open('data/mfcc_features/GE', 'rb'))
        bp = pickle.load(open('data/mfcc_features/BP', 'rb'))
        hi = pickle.load(open('data/mfcc_features/HI', 'rb'))
        sp = pickle.load(open('data/mfcc_features/SP', 'rb')) 
        '''
        
        
        '''
        print '**** delta features ****'
        ta = pickle.load(open('data/delta_features/TA', 'rb'))
        ge = pickle.load(open('data/delta_features/GE', 'rb'))
        bp = pickle.load(open('data/delta_features/BP', 'rb'))
        hi = pickle.load(open('data/delta_features/HI', 'rb'))
        sp = pickle.load(open('data/delta_features/SP', 'rb')) 
        '''
        '''
        print '**** FBANK features ****'
        ta = pickle.load(open('data/fbank_features/TA', 'rb'))
        ge = pickle.load(open('data/fbank_features/GE', 'rb'))
        bp = pickle.load(open('data/fbank_features/BP', 'rb'))
        hi = pickle.load(open('data/fbank_features/HI', 'rb'))
        sp = pickle.load(open('data/fbank_features/SP', 'rb')) 
        '''
        
        print "Filling the data set"
        self.X_all, self.y_all = [], []
        self.X_train, self.y_train = [], []
        self.X_test, self.y_test = [], []
        
        self.fill100 = False ####TODO 100 samples
        if (self.fill100 == False):
            print '*** all samples'
            ### use all samples available and then split 80/20
           
            self.fill(ar, 'AR')
            self.fill(ca, 'CA')
            self.fill(fr, 'FR')
            self.fill(ja, 'JA')
            self.fill(sp, 'SP')
            '''
           
            self.fill(ta, 'TA')
            self.fill(ge, 'GE')
            self.fill(bp, 'BP')
            self.fill(hi, 'HI')
            self.fill(sp, 'SP')
            '''
            ###
            
        else:
            print '***100 samples only'
            #### use only 100 samples and then split 80/20
            '''
            self.fillOnly100samples(ar, 'AR')
            self.fillOnly100samples(ca, 'CA')
            self.fillOnly100samples(fr, 'FR')
            self.fillOnly100samples(ja, 'JA')
            self.fillOnly100samples(sp, 'SP')
            '''
            
            self.fillOnly100samples(ta, 'TA')
            self.fillOnly100samples(ge, 'GE')
            self.fillOnly100samples(bp, 'BP')
            self.fillOnly100samples(hi, 'HI')
            self.fillOnly100samples(sp, 'SP')
            
        # convert data to meet sklearn interface
        self.X_all = np.array(self.X_all)
        self.y_all = np.array(self.y_all)
        print "X_all.shape", self.X_all.shape, "y_all.shape", self.y_all.shape

        if sk_split:
            # split inot train and test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_all, self.y_all, test_size=self.test_split)
            assert self.y_train.shape == (self.X_train.shape[0],)
            assert self.y_test.shape == (self.X_test.shape[0],)
        else:
            self.X_train = np.concatenate(self.X_train, axis=0)
            self.y_train = np.concatenate(self.y_train, axis=0)
            self.X_train, self.y_train = shuffle_data(self.X_train, self.y_train)
            print "X_train.shape", self.X_train.shape, "y_train.shape", self.y_train.shape

            self.X_test = np.concatenate(self.X_test, axis=0)
            self.y_test = np.concatenate(self.y_test, axis=0)
            self.X_test, self.y_test = shuffle_data(self.X_test, self.y_test)
            print "X_test.shape", self.X_test.shape, "y_test.shape", self.y_test.shape

        print 
        print "All data breakdown", len(self.X_all)
       
        print "    arabic =", np.sum(self.y_all == Label.arabic)
        print "    cantonese =", np.sum(self.y_all == Label.cantonese)
        print "    french =", np.sum(self.y_all == Label.french)
        print "    japanese =", np.sum(self.y_all == Label.japanese)
        print "    spanish =", np.sum(self.y_all == Label.spanish)
      
        '''
        print "    Tamil =", np.sum(self.y_all == Label.tamil)
        print "    Germany =", np.sum(self.y_all == Label.germany)
        print "    Brazilian P =", np.sum(self.y_all == Label.brazilianP)
        print "    Hindi =", np.sum(self.y_all == Label.hindi)
        print "    spanish =", np.sum(self.y_all == Label.spanish)
        '''

    def get_train_data(self, flatten=True):
        X_train = self.X_train
        if flatten:
            X_train = np.reshape(X_train, (X_train.shape[0], -1))
        return X_train, self.y_train

    def get_test_data(self, flatten=True):
        X_test = self.X_test
        if flatten:
            X_test = np.reshape(X_test, (X_test.shape[0], -1))
        return X_test, self.y_test

    def fill(self, data, label):
        X_batch, y_batch = [], []
        for sample, features in data.items():
            # ignore bad data
            if features.shape[0] < Config.min_seq_length:
                print "bad: y =", label, "shape =", features.shape
                continue

            
            
            '''
            if label == "TA": y = Label.tamil
            elif label == "GE": y = Label.germany
            elif label == "BP": y = Label.brazilianP
            elif label == "HI": y = Label.hindi
            elif label == "SP": y = Label.spanish
            else: continue
            '''
            if label == "AR": y = Label.arabic
            elif label == "CA": y = Label.cantonese
            elif label == "FR": y = Label.french
            elif label == "JA": y = Label.japanese
            elif label == "SP": y = Label.spanish
            else: continue

            if self.seq_length is None:
                self.seq_length, self.embed_size = features.shape
                print "seq_length", self.seq_length, "embed_size", self.embed_size
        
            self.X_all.append(features)
            self.y_all.append(y)

            if not self.sk_split:
                X_batch.append(features)
                y_batch.append(y)

        if not self.sk_split:
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            X_batch, y_batch = shuffle_data(X_batch, y_batch)

            limit = int((1.0 - self.test_split) * len(data))
            self.X_train.append(X_batch[:limit])
            self.y_train.append(y_batch[:limit])
            self.X_test.append(X_batch[limit:])
            self.y_test.append(y_batch[limit:])
            
    def fillOnly100samples(self, data, label): ### only fill with the first 100 samples for each language
        X_train_batch, y_train_batch = [], []
        X_test_batch, y_test_batch = [], []
        i = 0      
        for sample, features in data.items():
            if i == 100: break
            if i < 80: # training set 
                ## ignore bad data
                #print 'here', i
                if features.shape[0] < Config.min_seq_length:
                   print "bad: y =", label, "shape =", features.shape
                   continue
                
                
                '''
                if label == "TA": y = Label.tamil
                elif label == "GE": y = Label.germany
                elif label == "BP": y = Label.brazilianP
                elif label == "HI": y = Label.hindi
                elif label == "SP": y = Label.spanish
                else: continue
                '''
                if label == "AR": y = Label.arabic
                elif label == "CA": y = Label.cantonese
                elif label == "FR": y = Label.french
                elif label == "JA": y = Label.japanese
                elif label == "SP": y = Label.spanish
                else: continue
                #print 'there'

                X_train_batch.append(features)
                y_train_batch.append(y)

            else: ### test set
                
                X_test_batch.append(features)
                y_test_batch.append(y)
                
            self.X_all.append(features)
            self.y_all.append(y)
                
            if self.seq_length is None:
                    self.seq_length, self.embed_size = features.shape
                    print "seq_length", self.seq_length, "embed_size", self.embed_size        
            i+= 1

        X_train_batch = np.array(X_train_batch)
        y_train_batch = np.array(y_train_batch)
        X_train_batch, y_train_batch = shuffle_data(X_train_batch, y_train_batch)

        X_test_batch = np.array(X_test_batch)
        y_test_batch = np.array(y_test_batch)
        X_test_batch, y_test_batch = shuffle_data(X_test_batch, y_test_batch)

        self.X_train.append(X_train_batch)
        self.y_train.append(y_train_batch)
        self.X_test.append(X_test_batch)
        self.y_test.append(y_test_batch)
           

no_shuffle = False
def shuffle_data(X, y):
    if no_shuffle:
        return X, y
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def verify_accuracy(labels, actual):
    assert len(labels) == len(actual)
    a, c, f, j, s = 0, 0, 0, 0, 0
    
    total_a = float(np.sum(actual == Label.arabic))
    total_c = float(np.sum(actual == Label.cantonese))
    total_f = float(np.sum(actual == Label.french))
    total_j = float(np.sum(actual == Label.japanese))
    total_s = float(np.sum(actual == Label.spanish))
    '''
    total_a = float(np.sum(actual == Label.tamil))
    total_c = float(np.sum(actual == Label.germany))
    total_f = float(np.sum(actual == Label.brazilianP))
    total_j = float(np.sum(actual == Label.hindi))
    total_s = float(np.sum(actual == Label.spanish))
    '''
    
    total = total_a + total_c + total_f + total_j + total_s
    assert total == float(len(actual))
    
    for i in xrange(len(labels)):
        if labels[i] == actual[i]:
            
           
            '''
            if labels[i] == Label.tamil: a += 1
            elif labels[i] == Label.germany: c += 1
            elif labels[i] == Label.brazilianP: f += 1
            elif labels[i] == Label.hindi: j += 1
            elif labels[i] == Label.spanish: s += 1
            '''
            if labels[i] == Label.arabic: a += 1
            elif labels[i] == Label.cantonese: c += 1
            elif labels[i] == Label.french: f += 1
            elif labels[i] == Label.japanese: j += 1
            elif labels[i] == Label.spanish: s += 1
            else:
                raise Exception("unknown label %d" % labels[i])

    print "Arabic: %6.4f" % (a / safe_denominator(total_a))
    print "Cantonese: %6.4f" % (c / safe_denominator(total_c))
    print "French: %6.4f" % (f / safe_denominator(total_f))
    print "Japanese: %6.4f" % (j / safe_denominator(total_j))
    print "Spanish: %6.4f" % (s / safe_denominator(total_s))
    '''
    print "tamil: %6.4f" % (a / safe_denominator(total_a))
    print "germany: %6.4f" % (c / safe_denominator(total_c))
    print "brazilianP: %6.4f" % (f / safe_denominator(total_f))
    print "Hindi: %6.4f" % (j / safe_denominator(total_j))
    print "Spanish: %6.4f" % (s / safe_denominator(total_s))
    '''
    print "Total correct: %6.4f" % ((a+c+f+j+s) / total)
   
def safe_denominator(a):
    return a if a != 0.0 else 1.0
 
