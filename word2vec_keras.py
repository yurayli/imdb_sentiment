# Train the word2vec skip-gram under negative sampling with Keras
# Pre-processing using keras' tokenizer and skipgram data generator

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense
from keras.layers import merge
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from time import time


# import data
ul_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', header=0, quoting=3)
train = pd.read_csv('labeledTrainData.tsv', sep='\t')
test = pd.read_csv('testData.tsv', sep='\t')


# Tokenizing texts
tokenizer = Tokenizer(nb_words=5000)
tokenizer.fit_on_texts(test['review'])
vocab_size = len(tokenizer.word_index)
word_to_idx = tokenizer.word_index
idx_to_word = {i: w for w,i in word_to_idx.iteritems()}
revs = tokenizer.texts_to_sequences(test['review'])
revs = [np.array(rev)-1 for rev in revs]  # modify word index starting from index 0


# Load pre-trained model (GloVe)
GLOVE_DIR = 'glove/'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print "Found %s word vectors." % len(embeddings_index)
print "%s words in the vocabulary not in pre-trained model." % \
    sum([not embeddings_index.has_key(idx_to_word[i]) for i in range(1,vocab_size-1)])

def create_emb(emb_dim):
    emb = np.zeros((vocab_size, emb_dim))

    for i in range(1, vocab_size-1):
        word = idx_to_word[i]
        if embeddings_index.has_key(word):
            emb[i] = embeddings_index[word]
        else:
            # If we can't find the word in glove, randomly initialize
            emb[i] = normal(scale=0.05, size=(emb_dim,))

    # This is our "rare word" id - we want to randomly initialize
    emb[-1] = normal(scale=0.1, size=(emb_dim,))
    return emb

emb = create_emb(50)

# Build model
def build_sg():
    # Building word2vec model with negative sampling
    embedding_dim = 50
    # inputs
    w_input = Input(shape=(1, ), dtype='int32')
    word_embeddings = Embedding(vocab_size, embedding_dim, input_length=1)
    w_emb = word_embeddings(w_input)
    # context
    c_input = Input(shape=(1, ), dtype='int32')
    context_embeddings = Embedding(vocab_size, embedding_dim, input_length=1)
    c_emb = context_embeddings(c_input)
    output = merge([w_emb, c_emb], mode='dot', dot_axes=2)
    output = Reshape((1,), input_shape=(1, 1))(output)
    output = Activation('sigmoid')(output)
    return Model([w_input, c_input], output)

skip_gram = build_sg()
skip_gram.compile(loss='binary_crossentropy', optimizer='adam')
skip_gram.summary()


# Training
nb_epochs = 5
t_start = time()
t0 = t_start
for epoch in range(nb_epochs):
    print "Epoch %d:" %(epoch+1)
    for i, review in enumerate(revs):
        # skipgram function helps build data/labels with positive/negative smaples
        data, labels = skipgrams(sequence=review, vocabulary_size=vocab_size,
            window_size=2, negative_samples=5.)
        x = [np.array(x) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)
        #skip_gram.fit(x, y, batch_size=256, nb_epoch=1)
        loss = skip_gram.train_on_batch(x, y)
        if i%1000 == 0:
            print "training %d, with loss %.6f, elapsed time %.3fs." %(i, loss, time()-t0)
            t0 = time()
            #print "\n-------training %d, elapsed time %.3fs.\n" %(i, time()-t0)
    print "Loss %d, and elapsed time %.3fs for the epoch." % (loss, time()-t_start)


# See the results
embeddings = skip_gram.get_weights()[0]
contexts = skip_gram.get_weights()[1]
embedding_norm = np.sqrt(np.sum(embeddings**2, 1))
context_norm = np.sqrt(np.sum(contexts**2, 1))
normalized_embedding = embeddings / embedding_norm[:, None]
normalized_context = contexts / context_norm[:, None]

sim_house = normalized_embedding.dot(normalized_embedding[word_to_idx['house']-1])
nearst = (-sim_house).argsort()[:10]
print "The nearst 10 words to 'house':", [idx_to_word[i] for i in nearst]
rel_house = normalized_context.dot(normalized_embedding[word_to_idx['house']-1])
most_related = (-rel_house).argsort()[:10]
print "The most related 10 words to 'house':", [idx_to_word[i] for i in most_related]
