## imdb movie reviews: train a sentiment classifier using recurrent network
# CV auc score: 0.949375
# test auc score: 0.954490

# load libraries
import os
import re
import itertools
import cPickle as pickle
from bs4 import BeautifulSoup
from time import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, merge
from keras.layers import Embedding, SimpleRNN, LSTM, GRU
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping


path = "/input/"
output_path = "/output/"


def load_data(path):
    print "Loading data...\n"
    # import data
    #ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
    train = pd.read_csv(path + 'labeledTrainData.tsv', sep='\t')
    test = pd.read_csv(path + 'testData.tsv', sep='\t')
    return train, test


def tokenize(train, test, vocab_size):
    # Clean and tokenize the review sentences as input
    print "Cleaning and tokenizing the review texts..."
    tokenizer = Tokenizer(nb_words=vocab_size)
    tokenizer.fit_on_texts(train['review'])
    train_tokens = tokenizer.texts_to_sequences(train['review'])
    test_tokens = tokenizer.texts_to_sequences(test['review'])
    return train_tokens, test_tokens


def build_model(vocab_size, seq_len):  # ~9x second/epoch on local cpu machine
    inp = Input(shape=(seq_len,), dtype='int32', name='model_input')
    emb = Embedding(vocab_size+1, 16, input_length=seq_len, mask_zero=True)(inp)
    x = GRU(32)(emb)
    x = Dropout(0.25)(x)
    x = Dense(1, activation='sigmoid')(x)
    net = Model(inp, x)
    net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return net


def train_rnn(train_tokens, train_targets, test_tokens, vocab_size, seq_len=500, nb_folds=5):
    train_tensor = sequence.pad_sequences(train_tokens, maxlen=seq_len, value=0)
    test_x = sequence.pad_sequences(test_tokens, maxlen=seq_len, value=0)
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=0)
    cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train_tensor)]
    preds_val = []
    preds_test = []
    for i in range(nb_folds):
        print "\nTraining fold %d..." %(i+1)
        train_x, val_x = train_tensor[cv_indices[i][0]], train_tensor[cv_indices[i][1]]
        train_y, val_y = train_targets[cv_indices[i][0]], train_targets[cv_indices[i][1]]
        net = build_model(vocab_size, seq_len)
        stopping = EarlyStopping(min_delta=0, patience=1)
        net.fit(train_x, train_y, validation_data=(val_x, val_y),
            nb_epoch=2, batch_size=64, callbacks=[stopping])
        preds_val.append(net.predict(val_x).ravel())
        preds_test.append(net.predict(test_x).ravel())
    print "CV score: %.4f" \
        %(np.mean([roc_auc_score(train_targets[cv_indices[i][1]], preds_val[i]) for i in range(nb_folds)]))
    return np.mean(preds_test, 0)


def run(vocab_size=5000):
    train, test = load_data(path)
    train_tokens, test_tokens = tokenize(train, test, vocab_size)
    test_id = test['id']
    train_targets = train['sentiment'].values
    del train, test
    pred = train_rnn(train_tokens, train_targets, test_tokens, vocab_size)
    submit = pd.DataFrame({'id': test_id, 'sentiment': pred})
    submit.to_csv(output_path + 'imdb_rnn.csv', index=False)


if __name__ == "__main__":
    run()

