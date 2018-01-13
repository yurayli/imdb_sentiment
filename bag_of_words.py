## imdb movie reviews: train a sentiment classifier using bag of words

# load libraries
import os
import re
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from nltk.corpus import stopwords

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout


# import data
def load_data():
    print "Loading data...\n"
    ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
    train = pd.read_csv('labeledTrainData.tsv', sep='\t')
    test = pd.read_csv('testData.tsv', sep='\t')
    return train, test, ul_train


# text cleaning
def clean_text(raw, remove_stopwords=False):
    # remove punctuation and simplify numbers
    cleanedText = re.sub(r'[^\w\s\']+', '', raw)
    # change words to lowercase and split them
    words = cleanedText.lower().split()
    # remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]
    return ' '.join(words)


# train a vectorizer for test features
def vectorizer_fit(train, test, ul_train, num_features=10000, vec_type='tfidf'):
    # represent data with features
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    print "Creating the bag of words...\n"
    if vec_type=='tfidf':
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None,
            preprocessor=None, stop_words=None, max_features=num_features)
    else:
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
            preprocessor=None, stop_words=None, max_features=num_features)
    vectorizer.fit(pd.concat([train['review'], test['review'], ul_train['review']]))
    return vectorizer


def train_linear(train, test, vectorizer, vec_type='tfidf', nb_folds=5):
    print "Start training..."
    # train/val split
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=0)
    cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train)]
    preds_lr_val = []
    preds_lr_test = []
    preds_gbl_val = []
    preds_gbl_test = []
    test_x = vectorizer.transform(test['review'])
    if vec_type=='tfidf':
        clf = LogisticRegression(C=2, solver='sag')
        param = {'nthread':4, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc',
            'eta':0.2, 'alpha':0.5, 'lambda':0.02, 'booster':'gblinear'}
        num_round = 6
    else:
        clf = LogisticRegression(C=1e2, solver='sag', max_iter=500)
        param = {'nthread':4, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc',
            'eta':0.3, 'alpha':5., 'lambda':0., 'booster':'gblinear'}
        num_round = 5
    for i in range(nb_folds):
        print "Training fold %d..." %(i+1)
        train_x, val_x = train['review'][cv_indices[i][0]], train['review'][cv_indices[i][1]]
        train_y, val_y = train['sentiment'][cv_indices[i][0]], train['sentiment'][cv_indices[i][1]]
        train_x, val_x = vectorizer.transform(train_x), vectorizer.transform(val_x)
        clf.fit(train_x, train_y)
        preds_lr_val.append(clf.predict_proba(val_x)[:,1])
        preds_lr_test.append(clf.predict_proba(test_x)[:,1])
        dtrain = xgb.DMatrix(train_x, train_y)
        dval = xgb.DMatrix(val_x, val_y)
        dtest = xgb.DMatrix(test_x)
        seed = np.random.randint(100)
        param['seed'] = seed
        watchlist = [(dtrain, 'train'), (dval, 'validation')]
        bst = xgb.train(param, dtrain, num_round, watchlist)
        preds_gbl_val.append(bst.predict(dval))
        preds_gbl_test.append(bst.predict(dtest))
    print "CV score of logistic regression: %.4f" \
        %(np.mean([roc_auc_score(train['sentiment'][cv_indices[i][1]], preds_lr_val[i]) for i in range(nb_folds)]))
    print "CV score of gblinear booster: %.4f" \
        %(np.mean([roc_auc_score(train['sentiment'][cv_indices[i][1]], preds_gbl_val[i]) for i in range(nb_folds)]))
    return np.mean(preds_lr_test, 0), np.mean(preds_gbl_test, 0)


def train_nn(train, test, vectorizer, nb_folds=5):
    print "Start training..."
    # train/val split
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=99)
    cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train)]
    preds_nn_val = []
    preds_nn_test = []
    test_x = np.array(vectorizer.transform(test['review']).todense(), dtype='float32')
    for i in range(nb_folds):
        print "\nTraining fold %d..." %(i+1)
        train_x, val_x = train['review'][cv_indices[i][0]], train['review'][cv_indices[i][1]]
        train_y, val_y = train['sentiment'][cv_indices[i][0]], train['sentiment'][cv_indices[i][1]]
        train_x = np.array(vectorizer.transform(train_x).todense(), dtype='float32')
        val_x = np.array(vectorizer.transform(val_x).todense(), dtype='float32')
        net = Sequential([
                Dense(160, input_dim=val_x.shape[1], activation='tanh'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')])
        net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        net.fit(train_x, train_y, validation_data=[val_x, val_y], nb_epoch=2, batch_size=128)
        preds_nn_val.append(net.predict(val_x).ravel())
        preds_nn_test.append(net.predict(test_x).ravel())
    print "\nCV score of neural network: %.4f" \
        %(np.mean([roc_auc_score(train['sentiment'][cv_indices[i][1]], preds_nn_val[i]) for i in range(nb_folds)]))
    return np.mean(preds_nn_test, 0)


def run():
    train, test, ul_train = load_data()
    print "Cleaning and parsing the training set movie reviews...\n"
    t0 = time()
    ul_train['review'] = ul_train['review'].apply(clean_text)
    train['review'] = train['review'].apply(clean_text)
    test['review'] = test['review'].apply(clean_text)
    print "Elapsed time %.2f sec for cleaning data" %(time()-t0)
    tfidf_vectorizer = vectorizer_fit(train, test, ul_train)
    count_vectorizer = vectorizer_fit(train, test, ul_train, vec_type='count')
    del ul_train
    ptest_lr_tfidf, ptest_gbl_tfidf = train_linear(train, test, tfidf_vectorizer)
    _, ptest_gbl_count = train_linear(train, test, count_vectorizer, vec_type='count')
    #ptest = train_nn(train, test, tfidf_vectorizer)
    del train
    ptest = np.mean([ptest_lr_tfidf, ptest_gbl_tfidf, ptest_gbl_tfidf, ptest_gbl_count], 0)
    print "\nSaving the prediction results..."
    submit = pd.DataFrame({'id': test['id'], 'sentiment': ptest})
    submit.to_csv('imdb_bow.csv', index=False)


if __name__ == "__main__":
    run()