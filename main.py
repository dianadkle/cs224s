import numpy as np
import sys, random
from sklearn.decomposition import PCA
from feature_extractor import FeatureExtractor, verify_accuracy
from classifier import Classifier, combine_preds
from sklearn.neural_network import MLPClassifier


class Config:
    random_seed = 101
    seq_length = 50
    embed_size = 13


def main(argv):
    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)

    # feature extractor
    extractor = FeatureExtractor()
    X_train, y_train = extractor.get_train_data()
    X_test, y_test = extractor.get_test_data()

    # convert to PCA featurs
    pca_ndims = 20 #80 #20
    pca = PCA(pca_ndims).fit(X_train)
    X_pca_train = pca.transform(X_train)
    X_pca_test = pca.transform(X_test)

    # convert to MFCC-level PCA
    X_mfcc_pca_train, X_mfcc_pca_test, mfcc_pca_ndims = compute_mfcc_pca(extractor)

    # run classifers
    
    knn_preds = run_classifier("KNN (K=5)", "knn", False, X_train, y_train, X_test, y_test)
    
    ### can comment out svm when tuning NN
    ### '''
    svm_cs_preds = run_classifier("SVM (LinearSVC) crammer_singer", "svm", True, X_train, y_train, X_test, y_test, svm_multi_class="crammer_singer", svm_C=0.01)
    svm_cs_mfcc_preds = run_classifier("SVM (LinearSVC) MFCC PCA(4) crammer_singer", "svm", True, X_mfcc_pca_train, y_train, X_mfcc_pca_test, y_test, svm_multi_class="crammer_singer", svm_dual=False, svm_C=0.1)
    
    #print '*****PCA= ", pca_ndims
        
    svm_cs_pca_preds = run_classifier("SVM (LinearSVC) PCA crammer_singer", "svm", True, X_pca_train, y_train, X_pca_test, y_test, svm_multi_class="crammer_singer", svm_dual=False, svm_C=1.0)
    svm_cs_preds_list = [svm_cs_pca_preds, svm_cs_mfcc_preds, svm_cs_preds]

    svm_preds = run_classifier("SVM (LinearSVC)", "svm", True, X_train, y_train, X_test, y_test, svm_C=0.01)
    svm_mfcc_preds = run_classifier("SVM (LinearSVC) MFCC PCA(4)", "svm", True, X_mfcc_pca_train, y_train, X_mfcc_pca_test, y_test, svm_dual=False, svm_C=0.1)
    svm_pca_preds = run_classifier("SVM (LinearSVC) PCA", "svm", True, X_pca_train, y_train, X_pca_test, y_test, svm_dual=False, svm_C=1.0)
    svm_preds_list = [svm_pca_preds, svm_mfcc_preds, svm_preds]
    ### '''
    ### end of svm     
    
    
    
    nn_preds = run_classifier("NN (MLP)", "nn", False, X_train, y_train, X_test, y_test)
    
    #print self.classifier.score(X_test,y_test)
    
    #uncomment the following section to run lstm
    #lstm_preds = run_classifier("LSTM NN", "lstm", False, X_train, y_train, X_test, y_test)
    #lstm_mfcc_pca_preds = run_classifier("LSTM NN", "lstm", False, X_train, y_train, X_test, y_test, vector=(50, mfcc_pca_ndims))
    #nn_preds_list = [lstm_preds, nn_preds]
    
    nn_preds_list = [nn_preds]
    print
    print "**** combine ****"
    '''preds = combine_preds(svm_cs_preds_list + svm_preds_list + nn_preds_list + [knn_preds])'''
    # preds = combine_preds('''svm_cs_preds_list + svm_preds_list + '''nn_preds_list )
    preds = combine_preds(nn_preds_list )
    ### preds = combine_preds(svm_cs_preds_list + svm_preds_list + nn_preds + [knn_preds])
    verify_accuracy(preds, y_test)
    print

def run_classifier(title, name, normalize, X_train, y_train, X_test, y_test, svm_multi_class=None, svm_dual=True, svm_C=1.0, lstm_vector=None):
    print "**** svm_C = ", svm_C, "****"
    print "**** %s ****" % title
    classifier = Classifier(name, normalize=normalize, svm_multi_class=svm_multi_class, svm_dual=svm_dual, svm_C=svm_C, lstm_vector=lstm_vector)
    classifier.train(X_train, y_train) 
    accuracy, _ = classifier.test(X_train, y_train)
    print "Train Accuracy = %6.4f (data %snormalized)" % (accuracy, "" if normalize else "not ")
    accuracy, preds = classifier.test(X_test, y_test)
    print "Test Accuracy = %6.4f (data %snormalized)" % (accuracy, "" if normalize else "not ")
    verify_accuracy(preds, y_test) 
    '''if name ==    'nn':
          
            print '****** directly calling MLPclassifier'
            mlp = MLPClassifier(solver="adam", \
                                    random_state=Config.random_seed, \
                                    hidden_layer_sizes=classifier.hidden_layers, \
                                    max_iter=classifier.max_iter, \
                                    learning_rate_init=classifier.init_lr, \
                                    early_stopping=classifier.validation > 0, \
                                    validation_fraction=classifier.validation, learning_rate='adaptive', activation='logistic')
            mlp.fit(X_train, y_train)
            print '************score=',  mlp.score(X_test,y_test)
    '''
    return preds

def compute_mfcc_pca(extractor): # apply PCA to individual MFCC
    X_raw, _ = extractor.get_train_data(flatten=False)
    n_samples, seq_length, embed_size = X_raw.shape
    mfcc_pca_ndims = int(embed_size / 3)
    #print "mfcc_pca_ndims", mfcc_pca_ndims
    
    pca_list = [PCA(mfcc_pca_ndims).fit(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i in xrange(seq_length)]
    X_mfcc_pca_train = np.concatenate([pca.transform(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i, pca in enumerate(pca_list)], axis=1)
    assert X_mfcc_pca_train.shape[1] == seq_length * mfcc_pca_ndims

    X_raw, _ = extractor.get_test_data(flatten=False)
    X_mfcc_pca_test = np.concatenate([pca.transform(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i, pca in enumerate(pca_list)], axis=1)
    assert X_mfcc_pca_test.shape[1] == seq_length * mfcc_pca_ndims

    return X_mfcc_pca_train, X_mfcc_pca_test, mfcc_pca_ndims

def get_embed_size():
    return Config.embed_size

if __name__ == "__main__":
    main(sys.argv)
