import pprint
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
import datetime
import pdb
import pickle
import datetime
from collections import defaultdict
import pprint
import argparse
import difflib
import pickle
from sklearn.svm import SVR
import sklearn
import time
from nltk.corpus import wordnet_ic
from nltk import pos_tag

from nltk.corpus import wordnet as wn

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--trainfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--v', type=int, required=True)

vpath = "./data/vocab.txt"
V = 20000

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # V = 20002 # 10000 + unk + pad
        D = 300 # GoogleNews word2vec
        Cin = 1 # input channel
        ks = [2,3,4,5] # kernel size
        Cout = 5
        dropout = 0.2

        self.conv = nn.ModuleList([nn.Conv2d(Cin, Cout, (k, 2*D)).float() for k in ks])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(ks)*Cout, 1).float()
        # self.nn_l1 = nn.Linear(len(ks)*Cout, hn)
        # self.nn_reg = nn.Linear(hn, 1)

    def forward(self, input):
        # s1: batch_size x maxlen
        # x1 = self.embed(s1).double()
        # x2 = self.embed(s2).double()
        # input = torch.cat([x1, x2], 2) # batch_size x maxlen x 2D
        # input: N x maxlen x 2D
        input_unsq = input.unsqueeze(1)  # N x 1 x maxlen x 2D
        out = [F.relu(conv(input_unsq).squeeze(3)) for conv in self.conv]  # [(N x Cout x maxlen)] * len(ks)
        out = [F.max_pool1d(z, z.size(2)).squeeze(2) for z in out]  # [(N x Cout)] * len(ks)
        out = torch.cat(out, 1)  # N x len(ks)*Cout
        out = self.dropout(out).float()
        out = F.relu(out)
        out = self.fc(out).float()
        # out = self.nn_l1(out).float()
        # out = F.relu(out)
        # out = self.nn_reg(out).float()
        return out

def load_vocab(vocab, V):
    with open(vocab, 'r') as f:
        word2id, id2word = {}, {}
        cnt = 0
        for line in f.readlines()[:V]:
            pieces = line.split()
            if len(pieces) != 2:
                exit(-1)
            word2id[pieces[0]] = cnt
            id2word[cnt] = pieces[0]
            cnt += 1
    word2id['<UNK>'] = V
    word2id['<PAD>'] = V+1
    id2word[V] = '<UNK>'
    id2word[V+1] = '<PAD>'
    return word2id, id2word

def load_data(fname, w2id):
    """
    :return: data
            list of tuples (s1, s2, score)
            where s1 and s2 are list of index of words in vocab
    """
    def get_indxs(sentence, w2id):
        MAX_LEN = 25
        res = []
        sp = sentence.split()
        for word in sp:
            if word in w2id:
                res.append(w2id[word])
            else:
                res.append(V) # unk
        # pad/cut to MAX_LEN
        if len(res) > MAX_LEN:
            res = res[:MAX_LEN]
        else:
            res += [V+1]*(MAX_LEN-len(res))
        return res

    data = []
    score = []
    with open(fname, 'r')as f:
        cnt = 0
        for line in f:
            sp = line.split('\t')
            s1 = get_indxs(sp[0], w2id)
            s2 = get_indxs(sp[1], w2id)
            y = float(sp[2].strip())
            data.append((s1, s2, y, sp[0], sp[1]))
    return data

def load_data_pretrain_emb(fname):
    """return dict"""
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    res['<UNK>'] = np.random.random((300))
    res['<PAD>'] = np.random.random((300))
    return res

def load_example(data):
    random.shuffle(data)
    for i in range(len(data)):
        yield data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]

def mini_batch(data, batch_size):
    gen = load_example(data)
    while True:
        s1_lst, s2_lst, y_lst, s1s_lst, s2s_lst = [], [], [], [], []
        for i in range(batch_size):
            s1, s2, y, s1s, s2s = next(gen)
            s1_lst.append(s1)   # word indxes
            s2_lst.append(s2)
            s1s_lst.append(s1s) # sentences
            s2s_lst.append(s2s)
            y_lst.append(y)
        yield np.array(s1_lst), np.array(s2_lst), np.array(y_lst), s1s_lst, s2s_lst
def extract_emb(s1, s2, wrdvec, id2w):
    """return matrix of size MAXLEN x 2D"""
    def get_emb(s):
        res = []
        for sentence in s:
            tmp = []
            for wordid in sentence:
                word = id2w[wordid]
                if word in wrdvec:
                    tmp.append(wrdvec[word])
                else:
                    tmp.append(wrdvec['<UNK>'])
            res.append(tmp)
        return np.array(res)
    s1vec = get_emb(s1)
    s2vec = get_emb(s2)
    res = np.concatenate((s1vec, s2vec), axis=2)
    #res = s1vec + s2vec
    # pdb.set_trace()
    return res

def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    ss1 = s1.strip().split()
    ss2 = s2.strip().split()
    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    if len(ss1) + len(ss2) == 0:
        pdb.set_trace()
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .0)
    return score

def main(args):

    # load vocab
    w2id, id2w = load_vocab(vpath, V)

    # hyper param
    batch_size = 256
    lr = 0.005

    # load data
    data_train = load_data(args.trainfile, w2id)
    data_test = load_data(args.testfile, w2id)
    wrdvec = load_data_pretrain_emb('./data/pretrain-emb.pkl')
    cnn = CNN()

    # loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=lr, weight_decay=0.001)

    # train
    for epoch in range(30):
        cnn.train()
        for i, (s1, s2, score, s1s, s2s) in enumerate(mini_batch(data_train, batch_size)):
            input = Variable(torch.from_numpy(extract_emb(s1, s2, wrdvec, id2w))).float()
            score = Variable(torch.from_numpy(score)).float()
            optimizer.zero_grad()
            output = cnn(input)
            loss = criterion(output, score)
            loss.backward()
            optimizer.step()
            if (i) % 5 == 0:
                print(datetime.datetime.now(), 'Epoch {} batch {} loss: {}' .format(epoch, i, loss.data[0]))

        cnn.eval()
        res = []
        loss_val = 0
        for item in data_test:
            s1, s2, score, s1s, s2s = np.array([item[0]]), np.array([item[1]]), np.array([item[2]]), item[3], item[4]
            input = Variable(torch.from_numpy(extract_emb(s1, s2, wrdvec, id2w))).float()
            score = Variable(torch.from_numpy(score)).float()
            output = cnn(input)
            loss = criterion(output, score)
            loss_val += loss.data.cpu().numpy()[0]
            res.append(output.data.cpu().numpy()[0][0])
        print('test error: ', float(loss_val) / len(data_test))


    torch.save(cnn.state_dict(), "./data/cnn.mdl")
    # evaluate
    cnn.eval()
    res = []
    loss_val = 0
    for item in data_test:
        s1, s2, score, s1s, s2s = np.array([item[0]]), np.array([item[1]]), np.array([item[2]]), item[3], item[4]
        input = Variable(torch.from_numpy(extract_emb(s1, s2, wrdvec, id2w))).float()
        output = cnn(input)
        loss = criterion(output, score)
        loss_val += loss.data.cpu().numpy()[0]
        res.append(output.data.cpu().numpy()[0][0])
    print('test error: ', float(loss_val) / len(data_test))

    # write prediction to file
    with open(args.predfile, 'w') as f:
        for i in res:
            f.write(str(i) + '\n')

# --------------------
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


# =========== util func ==============
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


notin_cnt = [0]

def cnn_embedding(args):
    # load vocab
    w2id, id2w = load_vocab(vpath, V)
    wrdvec = load_data_pretrain_emb('./data/pretrain-emb.pkl')
    # load data
    data_train = load_data(args.trainfile, w2id)
    data_test = load_data(args.testfile, w2id)
    cnn = CNN()
    cnn.load_state_dict(torch.load("./data/cnn.mdl"))
    res = []
    cnt = 0
    for item in data_train:
        s1, s2, s1s, s2s = np.array([item[0]]), np.array([item[1]]), item[3], item[4]
        input = Variable(torch.from_numpy(extract_emb(s1, s2, wrdvec, id2w))).float()
        output = cnn(input)
        res.append(output.data.cpu().numpy()[0][0])
        cnt += 1
        if cnt % 500 == 0:
            print('end ', cnt)

    # write prediction to file
    with open("./data/train-cnn.txt", 'w') as f:
        for i in res:
            f.write(str(i) + '\n')

    for item in data_test:
        s1, s2, s1s, s2s = np.array([item[0]]), np.array([item[1]]), item[3], item[4]
        input = Variable(torch.from_numpy(extract_emb(s1, s2, wrdvec, id2w))).float()
        output = cnn(input)
        res.append(output.data.cpu().numpy()[0][0])
        cnt += 1
        if cnt % 500 == 0:
            print('end ', cnt)

    # write prediction to file
    with open("./data/test-cnn.txt", 'w') as f:
        for i in res:
            f.write(str(i) + '\n')


# =========== feature extraction ==============
def sentence_similarity_word_alignment(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet and ppdb """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        L = [synset.path_similarity(ss) for ss in synsets2]
        L_prime = L
        L = [l for l in L if l]
        # Check that the similarity could have been computed

        if L:
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count > 0: score /= count
    return score  # , ppdb_score

def sentence_similarity_simple_baseline(s1, s2, counts=None):
    def embedding_count(s):
        ret_embedding = defaultdict(int)
        for w in s.split():
            w = w.strip('?.,')
            ret_embedding[w] += 1
        return ret_embedding

    first_sent_embedding = embedding_count(s1)
    second_sent_embedding = embedding_count(s2)
    Embedding1 = []
    Embedding2 = []
    if counts:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w] * 1.0 / (counts[w] + 0.001))
            Embedding2.append(second_sent_embedding[w] * 1.0 / (counts[w] + 0.001))
    else:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w])
            Embedding2.append(second_sent_embedding[w])
    ret_score = 0
    if not 0 == sum(Embedding2):
        # https://stackoverflow.com/questions/6709693/calculating-the-similarity-of-two-lists
        # https://docs.python.org/3/library/difflib.html
        sm = difflib.SequenceMatcher(None, Embedding1, Embedding2)
        ret_score = sm.ratio() * 5
    return ret_score

# =================================
def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    ss1 = s1.strip().split()
    ss2 = s2.strip().split()
    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .0)
    return score


def extract_absolute_difference(s1, s2):
    """t \in {all tokens, adjectives, adverbs, nouns, and verbs}"""
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = abs(len(s1) - len(s2)) / float(len(s1) + len(s2))
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return [t1, t2, t3, t4, t5]


def extract_mmr_t(s1, s2):
    shorter = 1
    if (len(s1) > len(s2)):  shorter = 2

    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = (len(s1) + 0.001) / (len(s2) + 0.001)
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = (cnt1 + 0.001) / (cnt2 + 0.001)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = (cnt1 + 0.001) / (cnt2 + 0.001)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = (cnt1 + 0.001) / (cnt2 + 0.001)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = (cnt1 + 0.001) / (cnt2 + 0.001)

    if shorter == 2:
        t1 = 1 / (t1 + 0.001)
        t2 = 1 / (t2 + 0.001)
        t3 = 1 / (t3 + 0.001)
        t4 = 1 / (t4 + 0.001)
        t5 = 1 / (t5 + 0.001)

    return [t1, t2, t3, t4, t5]


def sentence_similarity_information_content(sentence1, sentence2):
    ''' compute the sentence similairty using information content from wordnet '''
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        L = []
        for ss in synsets2:
            try:
                L.append(synset.res_similarity(ss, brown_ic))
            except:
                continue
        if L:
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count > 0: score /= count
    return score

def load_cnn_score(fname):
    res = []
    with open(fname, 'r') as f:
        for line in f:
            res.append(float(line.strip()))
    return res

def svr(args):
    T0 = time.time()
    # ----------
    # training
    first_sents = []
    second_sents = []
    true_score = []

    with open(args.trainfile, 'r') as f:
        for line in f.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            gs = line_split[2]
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)
            true_score.append(gs)

    Counts_for_tf = defaultdict(int)

    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1

    train_cnn = load_cnn_score('./data/train-cnn.txt')

    feature_scores = []
    N = len(first_sents)
    T1 = time.time()
    for i in range(N):
        s1 = first_sents[i]
        s2 = second_sents[i]

        scores = [  sentence_similarity_simple_baseline(s1,s2, Counts_for_tf),
                    sentence_similarity_word_alignment(s1, s2)
                    , sentence_similarity_information_content(s1,s2)
                    , extract_overlap_pen(s1, s2)
                    , *extract_absolute_difference(s1, s2)
                    , *extract_mmr_t(s1, s2)
                    , train_cnn[i]
        ]
        # , extract_doc2vec_similarity(s1,s2, model_doc2vec)]
        # cosine similarity
        feature_scores.append(scores)
        if 0 == (i + 1) % int(N / 10): print("%.2f" % ((i + 1) * 1.0 / N * 100), "%" + "finished (", time.time() - T1,
                                             ")")

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feature_scores)
    X_features = scaler.transform(feature_scores)
    print("Elapsed time:", time.time() - T0, "(preprocessing)")
    # clf = LinearRegression(); clf.fit(X_features, true_score)
    clf = SVR()  # R1 uses default parameters as described in SVR documentation
    clf.fit(X_features, true_score)

    # -----------
    # predicting
    first_sents = []
    second_sents = []
    with open(args.testfile, 'r') as f_val:
        for line in f_val.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)

    test_cnn = load_cnn_score('./data/test-cnn.txt')

    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1

    feature_scores = []
    N = len(first_sents)
    T1 = time.time()
    for i in range(N):
        s1 = first_sents[i]
        s2 = second_sents[i]

        scores = [   sentence_similarity_simple_baseline(s1,s2, Counts_for_tf),
                    sentence_similarity_word_alignment(s1, s2)
                    , sentence_similarity_information_content(s1,s2)
                    , extract_overlap_pen(s1, s2)
                    , *extract_absolute_difference(s1, s2)
                    , *extract_mmr_t(s1, s2)
                    , test_cnn[i]
                    ]
        # cosine similarity
        feature_scores.append(scores)
        if 0 == (i + 1) % int(N / 10): print("%.2f" % ((i + 1) * 1.0 / N * 100), "%" + "finished (", time.time() - T1,")")

    X_features = scaler.transform(feature_scores)
    Y_pred_np = clf.predict(X_features)
    Y_pred_np = [min(5, max(0, p), p) for p in Y_pred_np]
    with open(args.predfile, 'w') as f_pred:
        for i in range(len(Y_pred_np)):
            f_pred.write(str(Y_pred_np[i]) + '\n')
    print("Elapsed time:", time.time() - T0)

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
    cnn_embedding(args)
    svr(args)