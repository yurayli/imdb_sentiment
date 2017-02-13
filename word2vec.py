## NLP part 2: train a word vector model

# load libraries
import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from time import time

# import data
ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
train = pd.read_csv('labeledTrainData.tsv', sep='\t')
#test = pd.read_csv('testData.tsv', sep='\t')


##
# Text cleaning helper function
def clean_text(raw, remove_stopwords=False):
	# remove HTML markup
	text = BeautifulSoup(raw).get_text()
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
print "Tokenizing the review texts..."
t0 = time()
train['review'] = train['review'].apply(
	lambda x: [clean_text(s) for s in nltk.sent_tokenize(x.decode('utf-8'))])
ul_train['review'] = ul_train['review'].apply(
	lambda x: [clean_text(s) for s in nltk.sent_tokenize(x.decode('utf-8'))])

tokenized_sents = []
for rev in train['review']:
	tokenized_sents += rev
for rev in ul_train['review']:
	tokenized_sents += rev
print "Elapsed time %.2f for tokenizing" % (time()-t0)


##
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 30   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(tokenized_sents, workers=num_workers, \
            size=num_features, min_count=min_word_count, \
            window=context, sample=downsampling, sg=1)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_30minwords_10context"
model.save(model_name)


##
# Load model
model = gensim.models.Word2Vec.load(model_name)

# Using the trained model
print "The size of the trained word2vec model:", model.syn0.shape
print "Test finding the word doesn't match others:\n"
print "'%s' differs from other words in 'man woman child kitchen'" \
	% (model.doesnt_match("man woman child kitchen".split()))

print "The most similar words to 'man' is\n", model.most_similar("man")
print "Similarity between 'man' and 'woman':", model.similarity('man', 'woman')
print "\nThe most similar words to 'king' is\n", model.most_similar("king")
print "\nThe most similar words to 'awful' is\n", model.most_similar("awful")

print "Word embedding of 'computer' is", model['computer']
print "Creating a index to word mapping..."
index2word = model.wv.index2word

##
# Method 1:
# Featurize each review in the IMDB dataset with average of embeddings
counter = 0
def featurizeVec(review, model):
	global counter
	if counter%1000 == 0:
		print "Featurizing review %d..." % counter
	# remove stopwords
	stops = set(stopwords.words('english'))
	words = [w for sent in review for w in sent if not w in stops]
	# featurizing
	nwords = 0
	featureVec = 0
	index2word = model.wv.index2word
	for w in words:
		if w in index2word:
			nwords += 1
			featureVec = featureVec + model[w]
	featureVec = featureVec / nwords
	counter = counter + 1
	return featureVec

'''
def getFeatureVecs(reviews, model, num_features):
	vecsArray = np.zeros((len(reviews), num_features), dtype='float32')
	for counter in xrange(len(reviews)):
		if counter%1000 == 0:
			print "Review %d of %d..." % (counter, len(reviews))
		vecsArray[counter] = featurizeVec(reviews[counter], model)
	return vecsArray

train_x = getFeatureVecs(train['review'], model, num_features)
'''

train_x = train['review'].apply(lambda x: featurizeVec(x, model)).values
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(
	train_x, train['sentiment'], test_size=0.1, random_state=0)

# Training a sentiment classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)
t0 = time()
clf.fit(train_x, train_y)
print "Elapsed time %.2f sec for training." %(time()-t0)


# Method 2:
# Featurize using the similarity of words in a related cluster (vector quantization)

