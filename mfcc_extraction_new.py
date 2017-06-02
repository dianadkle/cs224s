from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import pickle 

lang = "SP" # can change to other langs

mfcc_samples = {}
delta_samples = {}
fbank_samples = {}
i = 0
for filename in os.listdir("speech_samples/" + lang ):
	(rate, sig) = wav.read(os.path.join(os.getcwd(), "speech_samples/" + lang,filename))
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	mfcc_samples[i] = mfcc_feat[0:50, :]  # can change 50 to 200 or 100
	delta_samples[i] = d_mfcc_feat[0:50, :] # can change 50 to 200 or 100
	fbank_samples[i] = fbank_feat[0:50, :] # can change 50 to 200 or 100
	i+=1

pickle.dump(mfcc_samples, open("speech_samples/mfcc_features/" + lang, "wb"))
pickle.dump(delta_samples, open("speech_samples/delta_features/" + lang, "wb"))
pickle.dump(delta_samples, open("speech_samples/fbank_features/" + lang, "wb"))