import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor, verify_accuracy
#### from neural_network import LstmNeuralNetwork

class Config:
    random_seed = 101


class Classifier(object):
    def __init__(self, algo, normalize=False, svm_multi_class=None, svm_dual=True, svm_C=1.0, lstm_vector=None):
        self.algo = algo
        self.normalize = normalize

        if self.algo == "svm":
            if svm_multi_class is None:
                self.classifier = LinearSVC(dual=svm_dual, C=svm_C, random_state=Config.random_seed)
            else:
                self.classifier = LinearSVC(multi_class=svm_multi_class, dual=svm_dual, C=svm_C, random_state=Config.random_seed)
        
        elif self.algo == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        
        elif self.algo == "nn":
            self.hidden_layers = (256, 256, 256) #(256, 256, 256, 256, 100) # (256, 256, 256) 0.32 default (100,)
            #(256,256, 100, 100) 0.3371
            self.max_iter = 1000 # default 200
            self.init_lr = 0.0001 # default 0.001
            self.validation = 0 # default False, no early stopping
            self.classifier = MLPClassifier(solver="adam", \
                                    random_state=Config.random_seed, \
                                    hidden_layer_sizes=self.hidden_layers, \
                                    max_iter=self.max_iter, \
                                    learning_rate_init=self.init_lr, \
                                    early_stopping=self.validation > 0, \
                                    validation_fraction=self.validation
                                    #,learning_rate='adaptive'
                                    #, activation='logistic' 
                                    )
            
            print '**** random seed = ', Config.random_seed
            print '**** NN config:', self.hidden_layers, ',', self.max_iter, ',', self.init_lr, ',' , self.validation                                                                              
        elif self.algo == "lstm":
            self.classifier = LstmNeuralNetwork(vector=lstm_vector)

        else:
            raise Exception("unknown mode " + self.algo)

    def train(self, X, y):
        """ M samples (rows), N features (columns)
        X: each row is a feature vector of an example.
        y: each element is a label.
        X.shape[0] == y.shape[0]
        """
        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std

        self.classifier.fit(X, y)
        

    def test(self, X, y):
        if self.normalize:
            X = (X - self.mean) / self.std
        
        preds = self.classifier.predict(X)
        accuracy = self.classifier.score(X, y)
        return accuracy, preds

 
# join knn_preds, svm_preds, nn_preds
def combine_preds(preds_list):
    for preds in preds_list:
        assert len(preds) == len(preds_list[0])

    return majority_vote(preds_list)

def majority_vote(preds_list):
    preds_list = [np.reshape(preds, (-1, 1)) for preds in preds_list]
    preds_mat = np.concatenate(preds_list, axis=1)
    n_samples = preds_mat.shape[0]

    y = np.zeros(n_samples, dtype=preds_mat.dtype)
    for i in xrange(n_samples):
        # majority vote
        counts = np.bincount(preds_mat[i]) # cluster's votes
        y[i] = np.argmax(counts) # find the cluster with most votes

    return y

"""
def combine_preds(knn_preds, svm_preds, nn_preds):
    assert len(nn_preds) == len(svm_preds)
    assert len(nn_preds) == len(knn_preds)
    return majority_vote([nn_preds, svm_preds, knn_preds])

    preds = []
    for i in xrange(len(nn_preds)):
        y_knn, y_svm, y_nn = knn_preds[i], svm_preds[i], nn_preds[i]
    
        y = y_nn
        '''
        if y_nn == cantonese or y_nn == japanese:
            if y_svm == spanish or y_svm == japanese:
                y = y_svm
            # majority vote
            if y_nn == y_knn or y_nn == y_svm:
                y = y_nn
            elif y_svm == y_knn:
                y = y_svm
        '''
        # majority vote
        if y_nn == y_knn or y_nn == y_svm:
            y = y_nn
        elif y_svm == y_knn:
            y = y_svm
        preds.append(y)
    return preds
"""
