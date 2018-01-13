## imdb movie reviews: train a sentiment classifier using convolutional network

# load libraries
import os
import re
import itertools
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
from keras.layers import Embedding, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


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


def build_conv_1(vocab_size, seq_len):  # 3 epochs
    conv = Sequential()
    conv.add(Embedding(vocab_size, 64, input_length=seq_len, dropout=0.2))
    conv.add(Dropout(0.2))
    conv.add(Convolution1D(128, 5, border_mode='same', activation='relu'))
    conv.add(GlobalMaxPooling1D())
    conv.add(Dense(128, activation='relu'))
    conv.add(Dropout(0.75))
    conv.add(Dense(1, activation='sigmoid'))
    conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return conv


def build_conv_2(vocab_size, seq_len):  # 2 epochs
    conv = Sequential()
    conv.add(Embedding(vocab_size, 64, input_length=seq_len, dropout=0.2))
    conv.add(Dropout(0.2))
    conv.add(Convolution1D(48, 5, border_mode='same', activation='relu'))
    conv.add(MaxPooling1D(pool_length=4))
    conv.add(Flatten())
    conv.add(Dense(128, activation='relu'))
    conv.add(Dropout(0.75))
    conv.add(Dense(1, activation='sigmoid'))
    conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return conv


def build_conv_mix(vocab_size, seq_len):
    inp = Input(shape=(seq_len,), dtype='int32', name='model_input')
    emb = Embedding(vocab_size, 64, input_length=seq_len, dropout=0.2)(inp)
    x = Dropout(0.2)(emb)

    branch3 = Convolution1D(32, 3, border_mode='same', activation='relu')(x)
    branch5 = Convolution1D(64, 5, border_mode='same', activation='relu')(x)
    branch7 = Convolution1D(24, 7, border_mode='same', activation='relu')(x)
    x = merge([branch3, branch5, branch7], mode='concat', concat_axis=2)

    #x = GlobalMaxPooling1D()(x)  # 3 epochs
    x = MaxPooling1D(pool_length=4)(x)  # 2 epochs
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(1, activation='sigmoid')(x)
    conv = Model(inp, x)
    conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return conv


def build_model(vocab_size, seq_len):
    return build_conv_1(vocab_size, seq_len)


def train_cnn(train_tokens, train_targets, test_tokens, vocab_size, seq_len=500, nb_folds=5):
    train_tensor = sequence.pad_sequences(train_tokens, maxlen=seq_len, value=0)
    test_x = sequence.pad_sequences(test_tokens, maxlen=seq_len, value=0)
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=0)
    cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train_tensor)]
    preds_val = []
    preds_test = []
    for i in range(nb_folds):
        print "Training fold %d..." %(i+1)
        train_x, val_x = train_tensor[cv_indices[i][0]], train_tensor[cv_indices[i][1]]
        train_y, val_y = train_targets[cv_indices[i][0]], train_targets[cv_indices[i][1]]
        conv = build_model(vocab_size, seq_len)
        conv.fit(train_x, train_y, validation_data=(val_x, val_y), nb_epoch=3, batch_size=64)
        preds_val.append(conv.predict(val_x).ravel())
        preds_test.append(conv.predict(test_x).ravel())
        conv.save_weights(output_path + 'model_%d.h5' %(i+1))
    print "CV score: %.4f" \
        %(np.mean([roc_auc_score(train_targets[cv_indices[i][1]], preds_val[i]) for i in range(nb_folds)]))
    return np.mean(preds_test, 0)


def run(vocab_size=50000):
    train, test = load_data(path)
    train_tokens, test_tokens = tokenize(train, test, vocab_size)
    test_id = test['id']
    train_targets = train['sentiment'].values
    del train, test
    pred = train_cnn(train_tokens, train_targets, test_tokens, vocab_size)
    submit = pd.DataFrame({'id': test_id, 'sentiment': pred})
    submit.to_csv(output_path + 'imdb_cnn.csv', index=False)


if __name__ == "__main__":
    run()

