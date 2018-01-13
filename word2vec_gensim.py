## IMDB movie review: train a sentiment classifier through
## featurizing each review in the with average of word embeddings
## Or
## featurizing each review in the with sum of word embeddings + tfidf weighting (worse than the former case in my exp)

# load libraries
import os
import re
from time import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# import data
def load_data():
	print "Loading data...\n"
	ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
	train = pd.read_csv('labeledTrainData.tsv', sep='\t')
	test = pd.read_csv('testData.tsv', sep='\t')
	return train, test, ul_train

# Text cleaning
def clean_text(raw, remove_stopwords=False):
	# remove punctuation and simplify numbers
	cleanedText = re.sub(r"[^\w\s\']+", '', text)
	# change words to lowercase and split them
	words = cleanedText.lower().split()
	# remove stopwords
	if remove_stopwords:
		stops = set(stopwords.words('english'))
		words = [w for w in words if not w in stops]
	return words


def train_word2vec(train, test, ul_train, num_features):
	tokenized_sents = []
	for rev in train['review']:
		tokenized_sents += rev
	for rev in test['review']:
		tokenized_sents += rev
	for rev in ul_train['review']:
		tokenized_sents += rev
	# Import the built-in logging module and configure it so that Word2Vec
	# creates nice output messages
	import logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
		level=logging.INFO)
	# Set values for various parameters
	#num_features = 200    # Word vector dimensionality
	min_word_count = 30   # Minimum word count
	num_workers = 4       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	# Initialize and train the model (this will take some time)
	from gensim.models import word2vec
	print "Training word vector model..."
	model = word2vec.Word2Vec(tokenized_sents, workers=num_workers, \
		size=num_features, min_count=min_word_count, \
		window=context, sample=downsampling, sg=1)
	# If you don't plan to train the model any further, calling
	# init_sims will make the model much more memory-efficient.
	#model.init_sims(replace=True)
	return model


def test_word2vec(model):
	# Using the trained model
	print "The size of the trained word2vec model:", model.wv.syn0.shape
	print "Test finding the word doesn't match others:\n"
	print "'%s' differs from other words in 'man woman child kitchen'" \
		% (model.doesnt_match("man woman child kitchen".split()))
	print "The most similar words to 'man' is\n", model.most_similar("man")
	print "Similarity between 'man' and 'woman':", model.similarity('man', 'woman')
	print "\nThe most similar words to 'king' is\n", model.most_similar("king")
	print "\nThe most similar words to 'awful' is\n", model.most_similar("awful")


def tfidf_vectorize(vocabulary, train, test, ul_train):
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None,
		preprocessor=None, stop_words=None, vocabulary=vocabulary)
	trn_revs = train['review'].apply(lambda r: ' '.join([w for s in r for w in s if w in vocabulary]))
	tt_revs = test['review'].apply(lambda r: ' '.join([w for s in r for w in s if w in vocabulary]))
	ul_revs = ul_train['review'].apply(lambda r: ' '.join([w for s in r for w in s if w in vocabulary]))
	vectorizer.fit(pd.concat([trn_revs, tt_revs, ul_revs]))
	train_tfidf = np.array(vectorizer.transform(trn_revs).todense()).astype('float32')
	test_tfidf = np.array(vectorizer.transform(tt_revs).todense()).astype('float32')
	return train_tfidf, test_tfidf


def featurizeVec(review, model, vocabulary, tfidf_vec=None):
	words = [w for sent in review for w in sent if w in vocabulary]
	# featurizing
	if tfidf_vec:
		featureVec = 0
		for w in words:
			featureVec = featureVec + model[w] * tfidf_vec[vocabulary[w]]
		return featureVec
	nb_words = 0
	featureVec = 0
	for w in words:
		nb_words += 1
		featureVec = featureVec + model[w]
	featureVec = featureVec / nb_words
	return featureVec


def getFeatureVecs(reviews, model, num_features, vocabulary, tfidf_array=None):
	vecsArray = np.zeros((len(reviews), num_features), dtype='float32')
	if tfidf_array:
		for counter in xrange(len(reviews)):
			if counter%1000 == 0:
				print "Featurizing review %d of %d..." % (counter, len(reviews))
			vecsArray[counter] = featurizeVec(reviews[counter], model, tfidf_array[counter], vocabulary)
		return vecsArray
	for counter in xrange(len(reviews)):
		if counter%1000 == 0:
			print "Featurizing review %d of %d..." % (counter, len(reviews))
		vecsArray[counter] = featurizeVec(reviews[counter], model, vocabulary)
	return vecsArray


def train_model(train, test, vocabulary, nb_folds=5, train_tfidf=None, test_tfidf=None):
	print "Start training..."
	# train/val split
	kf = KFold(n_splits=nb_folds, shuffle=True, random_state=0)
	cv_indices = [(tr_id, val_id) for tr_id, val_id in kf.split(train)]
	preds_val = []
	preds_test = []
	if train_tfidf and test_tfidf:
		train_vecs = getFeatureVecs(train['review'], model, num_features, vocabulary, train_tfidf)
		test_x = getFeatureVecs(test['review'], model, num_features, vocabulary, test_tfidf)
	else:
		train_vecs = getFeatureVecs(train['review'], model, num_features, vocabulary)
		test_x = getFeatureVecs(test['review'], model, num_features, vocabulary)
	for i in range(nb_folds):
		print "Training fold %d..." %(i+1)
		train_x, val_x = train_vecs[cv_indices[i][0]], train_vecs[cv_indices[i][1]]
		train_y, val_y = train['sentiment'][cv_indices[i][0]], train['sentiment'][cv_indices[i][1]]
		clf = LogisticRegression(C=2, solver='sag')
		clf.fit(train_x, train_y)
		preds_val.append(clf.predict_proba(val_x)[:,1])
		preds_test.append(clf.predict_proba(test_x)[:,1])
	print "CV score: %.4f" \
		%(np.mean([roc_auc_score(train['sentiment'][cv_indices[i][1]], preds_val[i]) for i in range(nb_folds)]))
	return np.mean(preds_test, 0)


def run(vocab_size=5000, num_features=200, tfidf=False):
	train, test, ul_train = load_data()
	# Clean and tokenize the review sentences as input
	print "Text preprocessing..."
	train['review'] = train['review'].apply(
		lambda x: [clean_text(s) for s in nltk.sent_tokenize(x.decode('utf-8'))])
	test['review'] = test['review'].apply(
		lambda x: [clean_text(s) for s in nltk.sent_tokenize(x.decode('utf-8'))])
	ul_train['review'] = ul_train['review'].apply(
		lambda x: [clean_text(s) for s in nltk.sent_tokenize(x.decode('utf-8'))])
	model = train_word2vec(train, test, ul_train, num_features)
	print "Creating a index to word mapping..."
	idx_to_word = model.wv.index2word
	vocabulary = {idx_to_word[i]:i for i in range(vocab_size)}
	if tfidf:
		train_tfidf, test_tfidf = tfidf_vectorize(vocabulary, train, test, ul_train)
		del ul_train
		pred = train_model(train, test, vocabulary, nb_folds=5, train_tfidf, test_tfidf)
		submit = pd.DataFrame({'id': test['id'], 'sentiment': pred})
		submit.to_csv('imdb_w2v_tfidf.csv', index=False)
	else:
		del ul_train
		pred = train_model(train, test, vocabulary)
		submit = pd.DataFrame({'id': test['id'], 'sentiment': pred})
		submit.to_csv('imdb_w2v.csv', index=False)


if __name__ == "__main__":
	run()