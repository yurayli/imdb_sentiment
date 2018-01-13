## Sentiment analysis from IMDB movie review

Data can be downloaded from
https://www.kaggle.com/c/word2vec-nlp-tutorial/data

---
try several models:
1. bag-of-words/tf-idf + linear model
2. word2vec + linear model
3. convolutional network
4. recurrent network
5. nbsvm

The ensembled model can achieve auc score of 0.9766 (cnn models + nbsvm).

Ref.
1. ICLR15: http://arxiv.org/abs/1412.5335
2. https://github.com/vinhkhuc/kaggle-sentiment-popcorn

### License ###
The code in NBSVM is based on Grégoire Mesnil's work at [iclr15](https://github.com/mesnilgr/iclr15), thus are released under [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).