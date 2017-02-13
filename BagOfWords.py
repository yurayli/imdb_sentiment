## NLP part 1: train a sentiment classifier using bag of words

# load libraries
import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from time import time

# import data
#ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
train = pd.read_csv('labeledTrainData.tsv', sep='\t')
test = pd.read_csv('testData.tsv', sep='\t')

# text cleaning
def clean_text(raw):
	# remove HTML markup
	text = BeautifulSoup(raw).get_text()
	# remove punctuation and simplify numbers
	cleanedText = re.sub(r'[\d]+', 'num', re.sub(r'[^\w\s]+', '', text))
	# change words to lowercase and split them
	words = cleanedText.lower().split()
	# remove stopwords
	stops = set(stopwords.words('english'))
	cleanedWords = [w for w in words if not w in stops]
	return ' '.join(cleanedWords)

print "Cleaning and parsing the training set movie reviews...\n"
t0 = time()
#ul_train['review'] = ul_train['review'].apply(clean_text)
train['review'] = train['review'].apply(clean_text)
test['review'] = test['review'].apply(clean_text)
print "Elapsed time %.2f sec for cleaning data" %(time()-t0)

# represent data with features
from sklearn.feature_extraction.text import CountVectorizer
print "Creating the bag of words...\n"

vectorizer = CountVectorizer(analyzer="word",   \
                             tokenizer=None,    \
                             preprocessor=None, \
                             stop_words=None,   \
                             max_features=5000)

train_x = vectorizer.fit_transform(train['review']).toarray()
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(
	train_x, train['sentiment'], test_size=0.1, random_state=0)

# Take a look at the words in the vocabulary
print "Words in the vocabulary:\n"
vocab = vectorizer.get_feature_names()
word_to_index = dict([(w, i) for i, w in enumerate(vocab)])

# Sum up the counts of each vocabulary word
dist = np.sum(train_x, axis=0)
# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

# Training a sentiment classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=20)
t0 = time()
clf.fit(train_x, train_y)
print "Elapsed time %.2f sec for training." %(time()-t0)

