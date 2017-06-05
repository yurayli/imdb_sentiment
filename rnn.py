## NLP part 3: train a rnn model

# load libraries
import os
import re
import itertools
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from time import time
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.layers import SimpleRNN, LSTM, GRU
from keras.preprocessing import sequence
import keras.callbacks as kcb


# import data
ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
train = pd.read_csv('labeledTrainData.tsv', sep='\t')
test = pd.read_csv('testData.tsv', sep='\t')

##
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
print "Elapsed time %.2f seconds for cleaning.\n" % (time()-t0)  # about 1 minute

vocab_size = 10000
word_freq = nltk.FreqDist(itertools.chain(
	*pd.concat([train['review'], ul_train['review'], test['review']], ignore_index=True) ))
vocab_freq = word_freq.most_common(vocab_size-1)
idx_to_word = [w[0] for w in vocab_freq] + ['UNK']
word_to_idx = {w: i for i, w in enumerate(idx_to_word)}

print "Tokenizing the review texts..."
t0 = time()
train['review'] = train['review'].apply(
	lambda x: np.array([word_to_idx[w] if w in idx_to_word else vocab_size-1 for w in x]))
print "Elapsed time %.2f seconds for tokenizing.\n" % (time()-t0)


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
checkpointer = kcb.ModelCheckpoint(filepath="imdb_lstm.h5", monitor='val_acc', save_best_only=True, verbose=1)

lstm = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
    #BatchNormalization(),
    LSTM(128),
    Dense(1, activation='sigmoid')])
lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm.fit(train_x, train_y,
	validation_data=(val_x, val_y), nb_epoch=3, batch_size=64,
	callbacks=[metricRecords, checkpointer])

pred = lstm.predict(val_x, batch_size=128)

from sklearn.metrics import roc_auc_score
print 'AUC score:', roc_auc_score(val_y, pred)


########
def build_rnn(embedding_dim):
    inp = Input(shape=(1,), dtype='int32', name='model_input')
    emb = Embedding(vocab_size, embedding_dim, input_length=seq_len, dropout=0.2)(inp)
    x = SimpleRNN(64, activation='relu', inner_init='identity', return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = SimpleRNN(64, activation='relu', inner_init='identity')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    net = Model(inp, x)
    net.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return net

rnn = build_rnn(64)
metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath=model_path+"imdb_rnn.h5", monitor='val_acc',
                                   save_best_only=True, verbose=1)


def build_gru(embedding_dim):
    inp = Input(shape=(1,), dtype='int32', name='model_input')
    emb = Embedding(vocab_size, embedding_dim, input_length=seq_len, dropout=0.2)(inp)
    x = GRU(128, consume_less='gpu', dropout_U=0.2, dropout_W=0.2, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    #x = GRU(128, consume_less='gpu', dropout_U=0.2, dropout_W=0.2)(x)
    #x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    net = Model(inp, x)
    net.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return net

gru = build_gru(128)
metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath=model_path+"imdb_gru.h5", monitor='val_acc',
                                   save_best_only=True, verbose=1)
