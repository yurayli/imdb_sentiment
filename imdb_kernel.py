import os, time, json, re
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, sampler

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

SEED = 2019
path = '../input/word2vec-nlp-tutorial/'
output_path = './'
EMBEDDING_FILE_GV = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'
EMBEDDING_FILE_PR = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
EMBEDDING_FILE_FT = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=500,
                    help='maximum length of a question sentence')
parser.add_argument('--vocab-size', type=int, default=50000)
parser.add_argument('--n-splits', type=int, default=10,
                    help='splits of n-fold cross validation')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size during training')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epochs', type=int, default=4,
                    help='number of training epochs')
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

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def clean_text(raw):
    return clean_numbers(clean_punc(raw.lower()))


def word_idx_map(raw_comments, vocab_size):
    texts = []
    for c in raw_comments:
        texts.append(c.split())
    word_freq = nltk.FreqDist(itertools.chain(*texts))
    vocab_freq = word_freq.most_common(vocab_size-2)
    idx_to_word = ['<pad>'] + [word for word, cnt in vocab_freq] + ['<unk>']
    word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}

    return idx_to_word, word_to_idx


def tokenize(comments, word_to_idx, maxlen):
    '''
    Tokenize and numerize the comment sequences
    Inputs:
    - comments: pandas series with wiki comments
    - word_to_idx: mapping from word to index
    - maxlen: max length of each sequence of tokens

    Returns:
    - tokens: array of shape (data_size, maxlen)
    '''

    tokens = []
    for c in comments.tolist():
        token = [(lambda x: word_to_idx[x] if x in word_to_idx else word_to_idx['<unk>'])(w) \
                 for w in c.split()]
        if len(token) > maxlen:
            token = token[-maxlen:]
        else:
            token = [0] * (maxlen-len(token)) + token
        tokens.append(token)
    return np.array(tokens).astype('int32')


# Load pre-trained word vector
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def get_embedding(embedding_file, embedding_dim, word_to_idx, vocab_size):
    with open(embedding_file, encoding="utf8", errors='ignore') as f:
        embeddings_index = dict(get_coefs(*o.split(' ')) for o in f if len(o)>100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    nb_words = min(vocab_size, len(word_to_idx))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
    for word, i in word_to_idx.items():
        if i >= vocab_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


class IMDB_dataset(Dataset):

    def __init__(self, tokenized_reviews, targets=None):
        self.reviews = tokenized_reviews
        self.targets = targets if targets is None else targets[:,None]

    def __getitem__(self, index):
        review = self.reviews[index]
        if self.targets is not None:
            target = self.targets[index]
            return torch.LongTensor(review), torch.FloatTensor(target)
        else:
            return torch.LongTensor(review)

    def __len__(self):
        return len(self.reviews)


def prepare_loader(x, y=None, batch_size=256, train=True):
    data_set = IMDB_dataset(x, y)
    if train:
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(data_set, batch_size=batch_size)


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

    def __init__(self, model, optimizer, scheduler=None, checkpoint_name='toxic_comment'):
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

        # for floydhub metric graphs
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
        self.dropout_emb = nn.Dropout2d(0.25)

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

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.gru.named_parameters():
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

    for p in params_1: p.requires_grad = True
    for p in params_2: p.requires_grad = True
    for p in params_3: p.requires_grad = True

    optimizer = torch.optim.Adam(params=[{'params': params_1}])
    optimizer.add_param_group({'params':params_2})
    optimizer.add_param_group({'params':params_3})

    return model, optimizer


def load_and_preproc():
    train_df = pd.read_csv(path+'labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv(path+'testData.tsv', sep='\t')

    print('cleaning text...')
    t0 = time.time()
    train_df['review'] = train_df['review'].apply(clean_text)
    test_df['review'] = test_df['review'].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    return train_df, test_df


def tokenize_reviews(train_df, test_df):
    y_train = train_df[label_cols].values.astype('uint8')
    full_text = train_df['review'].tolist() + test_df['review'].tolist()

    print('tokenizing...')
    t0 = time.time()
    idx_to_word, word_to_idx = word_idx_map(full_text, args.vocab_size)
    x_train = tokenize(train_df['review'], word_to_idx, args.maxlen)
    x_test = tokenize(test_df['review'], word_to_idx, args.maxlen)
    print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))

    return x_train, y_train, x_test, word_to_idx


def train_val_split(train_x, train_y):
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x)]
    return cv_indices


def main(args):
    # load data
    train_df, test_df = load_and_preproc()
    train_seq, train_tars, x_test, word_to_idx = tokenize_reviews(train_df, test_df)

    # load pretrained embedding
    print('loading embeddings...')
    t0 = time.time()
    embed_mat_1 = get_embedding(EMBEDDING_FILE_GV, 300, word_to_idx, args.vocab_size)
    embed_mat_2 = get_embedding(EMBEDDING_FILE_PR, 300, word_to_idx, args.vocab_size)
    #embed_mat_3 = get_embedding(EMBEDDING_FILE_FT, 300, word_to_idx, args.vocab_size)
    embed_mat = np.mean([embed_mat_1, embed_mat_2], 0)
    print('loading complete in {:.0f} seconds.'.format(time.time()-t0))

    # training preparation
    train_preds = np.zeros((len(train_tars),1), dtype='float32') # matrix for the out-of-fold predictions
    test_preds = np.zeros((len(test_df),1), dtype='float32') # matrix for the predictions on the testset
    test_loader = prepare_loader(x_test, train=False)
    cv_indices = train_val_split(train_seq, train_tars)

    print()
    for i, (trn_idx, val_idx) in enumerate(cv_indices):
        print(f'Fold {i + 1}')

        # train/val split
        x_train, x_val = train_seq[trn_idx], train_seq[val_idx]
        y_train, y_val = train_tars[trn_idx], train_tars[val_idx]
        train_loader = prepare_loader(x_train, y_train, args.batch_size)
        val_loader = prepare_loader(x_val, y_val, train=False)

        # model setup
        ft_lrs = [args.lr/20, args.lr, args.lr]
        model, optimizer = model_optimizer_init(192, args.vocab_size, embed_mat)
        scheduler = OneCycleScheduler(optimizer, args.epochs, train_loader, max_lr=ft_lrs, moms=(.8, .7))
        solver = NetSolver(model, optimizer, scheduler)

        # train
        t0 = time.time()
        solver.train_one_cycle(loaders=(train_loader, val_loader), epochs=args.epochs)
        time_elapsed = time.time() - t0
        print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        # inference
        solver.model.eval()
        test_scores = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device=device, dtype=torch.long)
                score = torch.sigmoid(solver.model(x))
                test_scores.append(score.cpu().numpy())
        test_scores = np.concatenate(test_scores)

        train_preds[val_idx] = solver.val_scores
        test_preds += test_scores / args.n_splits

        print()

    # submit
    print(f'For whole train set, val auc score is {roc_auc_score(train_tars, train_preds)}')
    submit = pd.DataFrame(test_preds, columns=[label_cols])
    submit['id'] = test_df['id'].copy()
    submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)