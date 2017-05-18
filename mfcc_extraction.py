from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import pickle 

mfcc_samples = {}
delta_samples = {}
i = 0
for filename in os.listdir("speech_samples/AR"):
	(rate, sig) = wav.read(os.path.join(os.getcwd(), "speech_samples/AR",filename))
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	mfcc_samples[i] = mfcc_feat[0:50, :]
	delta_samples[i] = d_mfcc_feat[0:50, :]
	i+=1

pickle.dump(mfcc_samples, open("speech_samples/mfcc_features/AR", "wb"))
pickle.dump(delta_samples, open("speech_samples/delta_features/AR", "wb"))