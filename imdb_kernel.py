import os, time, json, re
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Sampler

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

SEED = 2019
path = '../input/word2vec-nlp-tutorial/'
output_path = './'
EMBEDDING_FILES = [
    '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',
    '../input/pickled-paragram-300-vectors-sl999/paragram_300_sl999.pkl',
    '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=500,
                    help='maximum length of a question sentence')
parser.add_argument('--vocab-size', type=int, default=50000)
parser.add_argument('--n-splits', type=int, default=10,
                    help='splits of n-fold cross validation')
parser.add_argument('--nb-models', type=int, default=2,
                    help='number of models (folds) to ensemble')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epochs', type=int, default=4)
args = parser.parse_args()

label_cols = 'sentiment'
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_punc(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_text(raw):
    return clean_punc(raw.lower()).strip()


def word_idx_map(raw_texts, vocab_size):
    def build_vocab(sentences):
        """
        :param sentences: list of list of words
        :return: dictionary of words and their count
        """
        vocab = {}
        for sentence in sentences:
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        return vocab

    def most_common_vocab(vocab, k):
        """
        :param vocab: dictionary of words and their count
        :k: former k words to return
        :return: list of k most common words
        """
        sorted_vocab = sorted([(cnt,w) for w,cnt in vocab.items()])[::-1]
        return [(w,cnt) for cnt,w in sorted_vocab][:k]

    texts = [c.split() for c in raw_texts]
    word_freq = build_vocab(texts)
    vocab_freq = most_common_vocab(word_freq, vocab_size)
    idx_to_word = ['<pad>'] + [word for word, cnt in vocab_freq] + ['<unk>']
    word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}

    return idx_to_word, word_to_idx


def tokenize(texts, word_to_idx, maxlen):
    '''
    Tokenize and numerize the text sequences
    Inputs:
    - texts: raw texts
    - word_to_idx: mapping from word to index
    - maxlen: max length of each sequence of tokens

    Returns:
    - tokens: array of shape (data_size, maxlen)
    '''

    def text_to_id(c, word_to_idx, maxlen):
        return [(lambda x: word_to_idx[x] if x in word_to_idx else word_to_idx['<unk>'])(w) \
                 for w in c.split()[-maxlen:]]

    return [text_to_id(c, word_to_idx, maxlen) for c in texts]


# Seed for randomness in pytorch
def seed_torch(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Load pre-trained word vector
def load_embeddings(path):
    with open(path,'rb') as f:
        emb_index = pickle.load(f)
    return emb_index

def get_embedding(embedding_file, word_to_idx, embedding_dim=300):
    print(f'loading {embedding_file}')
    embeddings_index = load_embeddings(embedding_file)

    all_embs = np.stack([emb for emb in embeddings_index.values() if len(emb)==embedding_dim])
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    nb_words = len(word_to_idx)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
    for word, i in word_to_idx.items():
        if i > nb_words: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = embeddings_index.get(word.upper())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = embeddings_index.get(word.title())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
    return embedding_matrix


class IMDB_dataset(Dataset):

    def __init__(self, tokenized_reviews, targets=None, split=None, maxlen=256):
        self.reviews = tokenized_reviews
        self.targets = targets if targets is None else targets[:,None]
        self.split = split
        assert self.split in {'train', 'valid', 'test'}
        self.maxlen = maxlen

    def __getitem__(self, index):
        review = self.reviews[index]
        if self.targets is not None:
            target = self.targets[index]
            return review, torch.FloatTensor(target)
        else:
            return review

    def __len__(self):
        return len(self.reviews)

    def get_lens(self):
        lengths = np.fromiter(
            ((min(self.maxlen, len(seq))) for seq in self.reviews),
            dtype=np.int32)
        return lengths

    def collate_fn(self, batch):
        """
        Collate function for sequence bucketing
        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of reviews, and targets
        """

        if self.split in ('train', 'valid'):
            reviews, targets = zip(*batch)
        else:
            reviews = batch

        lengths = [len(c) for c in reviews]
        maxlen = max(lengths)
        padded_reviews = []
        for i, c in enumerate(reviews):
            padded_reviews.append([0]*(maxlen - lengths[i])+c)

        if self.split in ('train', 'valid'):
            return torch.LongTensor(padded_reviews), torch.stack(targets)
        else:
            return torch.LongTensor(padded_reviews)


class BucketSampler(Sampler):

    def __init__(self, data_source, sort_lens, bucket_size=None, batch_size=1024, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_lens = sort_lens
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_lens)
        self.weights = None

        if not shuffle_data:
            self.index = self.prepare_buckets()
        else:
            self.index = None

    def set_weights(self, weights):
        assert weights >= 0
        total = np.sum(weights)
        if total != 1:
            weights = weights / total
        self.weights = weights

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_lens)
            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = self.prepare_buckets(indices)
        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_lens)

    def prepare_buckets(self, indices=None):
        lengths = - self.sort_lens
        assert self.bucket_size % self.batch_size == 0 or self.bucket_size == len(lengths)

        if indices is None:
            if self.shuffle:
                indices = shuffle(np.arange(len(lengths), dtype=np.int32))
                lengths = lengths[indices]
            else:
                indices = np.arange(len(lengths), dtype=np.int32)

        #  bucket iterator
        def divide_chunks(l, n):
            if n == len(l):
                yield np.arange(len(l), dtype=np.int32), l
            else:
                # looping till length l
                for i in range(0, len(l), n):
                    data = l[i:i + n]
                    yield np.arange(i, i + len(data), dtype=np.int32), data

        new_indices = []
        extra_batch_idx = None
        for chunk_index, chunk in divide_chunks(lengths, self.bucket_size):
            # sort indices in bucket by descending order of length
            indices_sorted = chunk_index[np.argsort(chunk)]

            batch_idxes = []
            for _, batch_idx in divide_chunks(indices_sorted, self.batch_size):
                if len(batch_idx) == self.batch_size:
                    batch_idxes.append(batch_idx.tolist())
                else:
                    assert extra_batch_idx is None
                    assert batch_idx is not None
                    extra_batch_idx = batch_idx.tolist()

            # shuffling batches within buckets
            if self.shuffle:
                batch_idxes = shuffle(batch_idxes)
            for batch_idx in batch_idxes:
                new_indices.extend(batch_idx)

        if extra_batch_idx is not None:
            new_indices.extend(extra_batch_idx)

        if not self.shuffle:
            self.original_indices = np.argsort(indices_sorted).tolist()
        return indices[new_indices]


def prepare_loader(x, y=None, batch_size=256, split=None):
    assert split in {'train', 'valid', 'test'}
    dataset = IMDB_dataset(x, y, split, args.maxlen)
    if split == 'train':
        sampler = BucketSampler(dataset, dataset.get_lens(),
                                bucket_size=batch_size*20, batch_size=batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=dataset.collate_fn)
    else:
        sampler = BucketSampler(dataset, dataset.get_lens(),
                                batch_size=batch_size, shuffle_data=False)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=dataset.collate_fn), sampler.original_indices


# one-cycle scheduler
class OneCycleScheduler(object):

    def __init__(self, optimizer, epochs, train_loader, max_lr=3e-3,
                 moms=(.95, .85), div_factor=25, sep_ratio=0.3, final_div=None):

        self.optimizer = optimizer

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
            self.init_lrs = [lr/div_factor for lr in self.max_lrs]
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
            self.init_lrs = [max_lr/div_factor] * len(optimizer.param_groups)

        self.final_div = final_div
        if self.final_div is None: self.final_div = div_factor*1e4
        self.final_lrs = [lr/self.final_div for lr in self.max_lrs]
        self.moms = moms

        self.total_iteration = epochs * len(train_loader)
        self.up_iteration = int(self.total_iteration * sep_ratio)
        self.down_iteration = self.total_iteration - self.up_iteration

        self.curr_iter = 0
        self._assign_lr_mom(self.init_lrs, [moms[0]]*len(optimizer.param_groups))

    def _assign_lr_mom(self, lrs, moms):
        for param_group, lr, mom in zip(self.optimizer.param_groups, lrs, moms):
            param_group['lr'] = lr
            param_group['betas'] = (mom, 0.999)

    def _annealing_cos(self, start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start-end)/2 * cos_out

    def step(self):
        self.curr_iter += 1

        if self.curr_iter <= self.up_iteration:
            pct = self.curr_iter / self.up_iteration
            curr_lrs = [self._annealing_cos(min_lr, max_lr, pct) \
                            for min_lr, max_lr in zip(self.init_lrs, self.max_lrs)]
            curr_moms = [self._annealing_cos(self.moms[0], self.moms[1], pct) \
                            for _ in range(len(self.optimizer.param_groups))]
        else:
            pct = (self.curr_iter-self.up_iteration) / self.down_iteration
            curr_lrs = [self._annealing_cos(max_lr, final_lr, pct) \
                            for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)]
            curr_moms = [self._annealing_cos(self.moms[1], self.moms[0], pct) \
                            for _ in range(len(self.optimizer.param_groups))]

        self._assign_lr_mom(curr_lrs, curr_moms)


# solver of model with validation
class NetSolver(object):

    def __init__(self, model, optimizer, scheduler=None, checkpoint_name='imdb'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_name = checkpoint_name

        self.model = self.model.to(device=device)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        self.best_val_loss = 0.
        self.best_val_auc = 0.
        self.loss_history = []
        self.val_loss_history = []
        self.auc_history = []
        self.val_auc_history = []

    def _save_checkpoint(self, epoch, l_val, a_val):
        torch.save(self.model.state_dict(),
            output_path+self.checkpoint_name+'_%.3f_%.3f_epoch_%d.pth.tar' %(l_val, a_val, epoch))
        checkpoint = {
            'optimizer': str(type(self.optimizer)),
            'scheduler': str(type(self.scheduler)),
            'epoch': epoch,
        }
        with open(output_path+'hyper_param_optim.json', 'w') as f:
            json.dump(checkpoint, f)


    def forward_pass(self, x, y):
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=dtype)
        scores = self.model(x)
        loss = F.binary_cross_entropy_with_logits(scores, y)
        return loss, torch.sigmoid(scores)


    def lr_range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        epochs = int(np.ceil(num_it/len(train_loader)))
        n_groups = len(self.optimizer.param_groups)

        if isinstance(start_lr, list) or isinstance(start_lr, tuple):
            if len(start_lr) != n_groups:
                raise ValueError("expected {} max_lr, got {}".format(n_groups, len(start_lr)))
            self.start_lrs = list(start_lr)
        else:
            self.start_lrs = [start_lr] * n_groups

        curr_lrs = self.start_lrs*1
        for param_group, lr in zip(self.optimizer.param_groups, curr_lrs):
            param_group['lr'] = lr

        n, lrs_log, loss_log = 0, [], []

        for e in range(epochs):
            self.model.train()
            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lrs_log.append(curr_lrs[-1])
                loss_log.append(loss.item())

                # update best loss
                if n == 0:
                    best_loss, n_best = loss.item(), n
                else:
                    if loss.item() < best_loss:
                        best_loss, n_best = loss.item(), n

                # update lr per iter
                n += 1
                curr_lrs = [lr * (end_lr/lr) ** (n/num_it) for lr in self.start_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, curr_lrs):
                    param_group['lr'] = lr

                # stopping condition
                if n == num_it or (stop_div and (loss.item() > 4*best_loss or torch.isnan(loss))):
                    break

        print('minimum loss {}, at lr {}'.format(best_loss, lrs_log[n_best]))
        return lrs_log, loss_log


    def train(self, loaders, epochs):
        train_loader, val_loader = loaders

        # start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            running_loss = 0.

            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            train_auc, train_acc, _ = self.check_auc(train_loader, num_batches=50)
            val_auc, val_acc, val_loss = self.check_auc(val_loader, save_scores=True)

            self.log_and_checkpoint(e, train_loss, val_loss, train_auc, val_auc)

            if self.scheduler is not None:
                self.scheduler.step()


    def train_one_cycle(self, loaders, epochs):
        train_loader, val_loader = loaders

        # start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            running_loss = 0.

            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)

                # update lr, mom per iter
                self.scheduler.step()

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            train_auc, train_acc, _ = self.check_auc(train_loader, num_batches=50)
            val_auc, val_acc, val_loss = self.check_auc(val_loader, save_scores=True)

            self.log_and_checkpoint(e, train_loss, val_loss, train_auc, val_auc, train_acc, val_acc)


    def log_and_checkpoint(self, e, train_loss, val_loss, train_auc, val_auc, train_acc, val_acc):
        # checkpoint and record/print metrics at epoch end
        self.loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.auc_history.append(train_auc)
        self.val_auc_history.append(val_auc)

        # metrics log
        print('{"metric": "AUC", "value": %.4f, "epoch": %d}' % (train_auc, e+1))
        print('{"metric": "Val. AUC", "value": %.4f, "epoch": %d}' % (val_auc, e+1))
        print('{"metric": "Acc", "value": %.4f, "epoch": %d}' % (train_acc, e+1))
        print('{"metric": "Val. Acc", "value": %.4f, "epoch": %d}' % (val_acc, e+1))
        print('{"metric": "Loss", "value": %.4f, "epoch": %d}' % (train_loss, e+1))
        print('{"metric": "Val. Loss", "value": %.4f, "epoch": %d}' % (val_loss, e+1))

        if e == 0:
            self.best_val_auc = val_auc
            self.best_val_loss = val_loss
        if val_auc > self.best_val_auc:
            print('updating best val auc...')
            self.best_val_auc = val_auc
        if val_loss < self.best_val_loss:
            print('updating best val loss...')
            self.best_val_loss = val_loss
        print()


    def check_auc(self, loader, num_batches=None, save_scores=False):
        self.model.eval()
        targets, scores, losses = [], [], []
        with torch.no_grad():
            for t, (x, y) in enumerate(loader):
                l, score = self.forward_pass(x, y)
                targets.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
                losses.append(l.item())
                if num_batches is not None and (t+1) == num_batches:
                    break

        targets = np.concatenate(targets)
        scores = np.concatenate(scores)
        if save_scores:
            self.val_scores = scores  # to access from outside

        auc = roc_auc_score(targets, scores)
        acc = accuracy_score(targets, [1 if s>=0.5 else 0 for s in scores])
        loss = np.mean(losses)

        return auc, acc, loss


# model
class EmbeddingLayer(nn.Module):

    def __init__(self, embed_dim, vocab_size, embed_matrix):
        super(EmbeddingLayer, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.emb.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
        self.dropout_emb = nn.Dropout2d(0.4)

    def forward(self, seq):
        emb = self.emb(seq)
        emb = self.dropout_emb(emb.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
        return emb

class RecurrentNet(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(RecurrentNet, self).__init__()
        # Init layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)

        for mod in (self.lstm, self.gru):
            for name, param in mod.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)

    def forward(self, seq):
        o_lstm, _ = self.lstm(seq)
        o_gru, _ = self.gru(o_lstm)
        return o_gru

class IMDBClassifier(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(IMDBClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim*4, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, seq):
        avg_pool = torch.mean(seq, 1)
        max_pool, _ = torch.max(seq, 1)
        x = torch.cat((avg_pool, max_pool), 1)
        out = self.fc(self.dropout(x))
        return out

class IMDBNet(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_size, embed_matrix):
        super(IMDBNet, self).__init__()
        # Init layers
        self.emb_layer = EmbeddingLayer(embed_dim, vocab_size, embed_matrix)
        self.rnns = RecurrentNet(embed_dim, hidden_dim)
        self.classifier = IMDBClassifier(hidden_dim, 1)

    def forward(self, seq):
        emb = self.emb_layer(seq)
        o_gru = self.rnns(emb)
        out = self.classifier(o_gru)

        return out

def model_optimizer_init(nb_neurons, vocab_size, embed_mat):
    model = IMDBNet(300, nb_neurons, vocab_size, embed_mat)

    params_1 = [p for p in model.emb_layer.parameters()]
    params_2 = [p for p in model.rnns.parameters()]
    params_3 = [p for p in model.classifier.parameters()]

    optimizer = torch.optim.Adam(params=[{'params': params_1}])
    optimizer.add_param_group({'params':params_2})
    optimizer.add_param_group({'params':params_3})

    return model, optimizer

def eval_model(model, data_loader, mode='test'):
    assert mode in ('val', 'test')
    model.eval()
    test_scores = []
    with torch.no_grad():
        for x in data_loader:
            if mode=='val': x = x[0]
            x = x.to(device)
            score = torch.sigmoid(model(x))[:,0]
            test_scores.append(score.cpu().numpy())
    return np.concatenate(test_scores)


def load_preproc_and_tokenize():
    train_df = pd.read_csv(path+'labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv(path+'testData.tsv', sep='\t')

    print('cleaning text...')
    t0 = time.time()
    train_df['review'] = train_df['review'].apply(clean_text)
    test_df['review'] = test_df['review'].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    y_train = train_df[label_cols].values.astype('uint8')
    full_text = train_df['review'].tolist() + test_df['review'].tolist()

    print('tokenizing...')
    t0 = time.time()
    idx_to_word, word_to_idx = word_idx_map(full_text, args.vocab_size)
    x_train = tokenize(train_df['review'], word_to_idx, args.maxlen)
    x_test = tokenize(test_df['review'], word_to_idx, args.maxlen)
    print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))

    return x_train, y_train, x_test, word_to_idx, test_df['id'].tolist()


def train_val_split(train_x, train_y, nb_models=args.nb_models):
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x, train_y)]
    if nb_models:
        return cv_indices[:nb_models]
    return cv_indices


def main(args):
    # load and tokenize data
    train_seq, train_tars, x_test, word_to_idx, test_id = load_preproc_and_tokenize()

    # load pretrained embedding
    print('loading embeddings...')
    t0 = time.time()
    embed_mat = np.mean(
        [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=0)
    print('loading complete in {:.0f} seconds.'.format(time.time()-t0))

    # training preparation
    val_preds = []      # for the out-of-fold predictions
    test_preds = []     # for the predictions on the testset
    oof_tars = []       # for the oof targets
    test_loader, test_original_indices = prepare_loader(x_test, split='test')
    cv_indices = train_val_split(train_seq, train_tars)

    print()
    for i, (trn_idx, val_idx) in enumerate(cv_indices):
        print(f'Fold {i + 1}')

        # train/val split
        x_train, x_val = [train_seq[i] for i in trn_idx], [train_seq[i] for i in val_idx]
        y_train, y_val = train_tars[trn_idx], train_tars[val_idx]
        train_loader = prepare_loader(x_train, y_train, args.batch_size, split='train')
        val_loader, val_original_indices = prepare_loader(x_val, y_val, split='valid')
        oof_tars.append(y_val)

        # model setup
        ft_lrs = [args.lr*0.08, args.lr, args.lr]
        model, optimizer = model_optimizer_init(160, args.vocab_size, embed_mat)
        scheduler = OneCycleScheduler(optimizer, args.epochs, train_loader, max_lr=ft_lrs, moms=(.8, .7))
        solver = NetSolver(model, optimizer, scheduler)

        # train
        t0 = time.time()
        solver.train_one_cycle(loaders=(train_loader, val_loader), epochs=args.epochs)
        time_elapsed = time.time() - t0
        print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        # inference
        test_scores = eval_model(solver.model, test_loader)[test_original_indices]

        val_preds.append(solver.val_scores[val_original_indices])
        test_preds.append(test_scores)

        print()

    oof_tars = np.concatenate(oof_tars)
    val_preds = np.concatenate(val_preds)
    print(f'For whole train set, val auc score is {roc_auc_score(oof_tars, val_preds)}')

    # make submission
    test_preds = np.mean(test_preds, 0)
    submit = pd.DataFrame(test_preds, columns=[label_cols])
    submit['id'] = test_id
    submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)