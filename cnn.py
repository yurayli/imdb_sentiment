## NLP part 4: train a cnn model

# load libraries
import os
import re
import itertools
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim
from bs4 import BeautifulSoup
from time import time
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
import keras.callbacks as kcb


# import data
ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
train = pd.read_csv('labeledTrainData.tsv', sep='\t')
test = pd.read_csv('testData.tsv', sep='\t')

## Preparing
# Text cleaning helper function
def clean_text(raw, remove_stopwords=False):
	# remove HTML markup
	text = BeautifulSoup(raw, 'lxml').get_text()
	# remove punctuation and simplify numbers
	cleanedText = re.sub(r'[\d]+', 'num', re.sub(r'[^\w\s]+', '', text))
	# change words to lowercase and split them
	words = cleanedText.lower().split()
	# remove stopwords
	if remove_stopwords:
		stops = set(stopwords.words('english'))
		words = [w for w in words if not w in stops]
	return words

# Tokenize the review sentences as input
print "Cleaning the review texts..."
t0 = time()
train['review'] = train['review'].apply(clean_text)
ul_train['review'] = ul_train['review'].apply(clean_text)
test['review'] = test['review'].apply(clean_text)
print "Elapsed time %.2f for cleaning\n" % (time()-t0)

vocab_size = 5000
word_freq = nltk.FreqDist(itertools.chain( 
	*pd.concat([train['review'], ul_train['review'], test['review']], ignore_index=True) ))
vocab_freq = word_freq.most_common(vocab_size-1)
idx_to_word = [w[0] for w in vocab_freq] + ['UNK']
word_to_idx = {w: i for i, w in enumerate(idx_to_word)}

print "Tokenizing the review texts..."
train['review'] = train['review'].apply(
	lambda x: np.array([word_to_idx[w] if w in idx_to_word else vocab_size-1 for w in x]))
print "Elapsed time %.2f for tokenizing\n" % (time()-t0)


## Training
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(
	train['review'], train['sentiment'], test_size=0.1, random_state=0)
test['review'] = test['review'].apply(
	lambda x: np.array([word_to_idx[w] if w in idx_to_word else vocab_size-1 for w in x]))

seq_len = 500
train_x = sequence.pad_sequences(train_x, maxlen=seq_len, value=0)
val_x = sequence.pad_sequences(val_x, maxlen=seq_len, value=0)

# callback function during training
class CallMetric(kcb.Callback):
    def on_train_begin(self, logs={}):
        self.best_acc = 0.0
        self.accs = []
        self.val_accs = []
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        if logs.get('val_acc') > self.best_acc:
            self.best_acc = logs.get('val_acc')
            print("\nThe BEST val_acc to date.")

metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath="imdb_cnn.h5", monitor='val_acc', save_best_only=True)

'''
model = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(val_x, val_y), nb_epoch=3, batch_size=64)

conv1 = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
    Dropout(0.2),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    Dropout(0.2),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])
conv1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv1.fit(train_x, train_y, validation_data=(val_x, val_y), nb_epoch=3, batch_size=64)

conv2 = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
    BatchNormalization(),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])
conv2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv2.fit(train_x, train_y, 
	validation_data=(val_x, val_y), nb_epoch=6, batch_size=64, 
	callbacks=[metricRecords, checkpointer])
'''
conv2 = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
    Dropout(0.2),
    #BatchNormalization(),
    Convolution1D(64, 3, border_mode='same', activation='relu'),
    Convolution1D(64, 3, border_mode='same', activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])
conv2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv2.fit(train_x, train_y, 
	validation_data=(val_x, val_y), nb_epoch=5, batch_size=64, 
	callbacks=[metricRecords, checkpointer])


