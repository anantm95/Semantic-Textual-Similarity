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
from nltk import word_tokenize, pos_tag

from nltk.corpus import wordnet as wn

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--trainfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--v', type=int, required=True)

vpath = "./data/vocab.txt"
V = 10000

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        V = 10002 # 10000 + unk + pad
        D = 128 # word embedding size
        Cin = 1 # input channel
        ks = [1,2,3,4,5,6] # kernel size
        Cout = 20
        dropout = 0.2

        self.embed = nn.Embedding(V, D)
        self.conv = nn.ModuleList([nn.Conv2d(Cin, Cout, (k, 2*D)).double() for k in ks])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(ks)*Cout+11, 1).float()

    def forward(self, s1, s2, baseline_features):
        # s1: batch_size x maxlen
        x1 = self.embed(s1).double()
        x2 = self.embed(s2).double()
        input = torch.cat([x1, x2], 2) # batch_size x maxlen x 2D
        input = input.unsqueeze(1)  # N x 1 x maxlen x 2D
        out = [F.relu(conv(input).squeeze(3)) for conv in self.conv]  # [(N x Cout x maxlen)] * len(ks)
        out = [F.max_pool1d(z, z.size(2)).squeeze(2) for z in out]  # [(N x Cout)] * len(ks)
        out = torch.cat(out, 1)  # N x len(ks)*Cout
        out = self.dropout(out).float()
        out = torch.cat([out, baseline_features], 1)
        out = self.fc(out).float()
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
    return word2id, id2word

def load_data(fname, w2id):
    """
    :return: data
            list of tuples (s1, s2, score)
            where s1 and s2 are list of index of words in vocab
    """
    def get_indxs(sentence, w2id):
        MAX_LEN = 30
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
    with open(fname, 'r') as f:
        for line in f:
            sp = line.split('\t')
            s1 = get_indxs(sp[0], w2id)
            s2 = get_indxs(sp[1], w2id)
            y = float(sp[2].strip())
            data.append((s1, s2, y, sp[0], sp[1]))
    return data

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

def extract_absolute_difference(s1, s2):
    """t \in {all tokens, adjectives, adverbs, nouns, and verbs}"""
    s1, s2 = s1.split(), s2.split()
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
    if(len(s1) > len(s2)):  shorter = 2

    s1, s2 = s1.split(), s2.split()
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = (len(s1)+0.001) / (len(s2) +0.001)
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = (cnt1 +0.001) / (cnt2 + 0.001)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = (cnt1 +0.001) / (cnt2+0.001)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = (cnt1 +0.001) / (cnt2 +0.001)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = (cnt1+ 0.001) / (cnt2 + 0.001)

    if shorter == 2:
        t1 = 1 / (t1 + 0.001)
        t2 = 1 / (t2 + 0.001)
        t3 = 1 / (t3 + 0.001)
        t4 = 1 / (t4 + 0.001)
        t5 = 1 / (t5 + 0.001)

    return [t1, t2, t3, t4, t5]

def extract_baseline_features(s1, s2):
    res = []
    for i in range(len(s1)):
        st1, st2 = s1[i], s2[i]
        if st1 == ' ' and st2 == ' ':
            res.append([0]*11)
            continue
        tmp = []
        tmp.append(extract_overlap_pen(st1, st2))
        tmp.extend(extract_absolute_difference(st1, st2))
        tmp.extend(extract_mmr_t(st1, st2))
        res.append(tmp)
    return np.array(res)

def main(args):

    # load vocab
    w2id, id2w = load_vocab(vpath, V)

    # hyper param
    batch_size = 512
    lr = 0.03

    # load data
    data_train = load_data(args.trainfile, w2id)
    data_test = load_data(args.testfile, w2id)
    cnn = CNN()

    # loss
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=lr)

    # train
    for epoch in range(20):
        for i, (s1, s2, score, s1s, s2s) in enumerate(mini_batch(data_train, batch_size)):
            baseline_features = extract_baseline_features(s1s, s2s)
            baseline_features = Variable(torch.from_numpy(baseline_features)).float()
            s1 = Variable(torch.from_numpy(s1))
            s2 = Variable(torch.from_numpy(s2))
            score = Variable(torch.from_numpy(score)).float()
            optimizer.zero_grad()
            output = cnn(s1, s2, baseline_features)
            loss = criterion(output, score)
            loss.backward()
            optimizer.step()
            if (i) % 5 == 0:
                print(datetime.datetime.now(), 'Epoch {} batch {} loss: {}' .format(epoch, i, loss.data[0]))
        cnn.eval()
        res = []
        cnt = 0
        loss_val = 0
        for item in data_test:
            s1, s2, score, s1s, s2s = np.array([item[0]]), np.array([item[1]]), np.array([item[2]]), item[3], item[4]
            baseline_features = extract_baseline_features([s1s], [s2s])
            baseline_features = Variable(torch.from_numpy(baseline_features)).float()
            score = Variable(torch.from_numpy(score)).float()
            s1 = Variable(torch.from_numpy(s1))
            s2 = Variable(torch.from_numpy(s2))
            output = cnn(s1, s2, baseline_features)
            loss = criterion(output, score)
            loss_val += loss.data.cpu().numpy()[0]
            res.append(output.data.cpu().numpy()[0][0])
        print('test error: ', float(loss_val) / len(data_test))
    torch.save(cnn.state_dict(), "./data/cnn.mdl")
    # evaluate
    cnn.eval()
    res = []
    cnt = 0
    for item in data_test:
        s1, s2, score, s1s, s2s = np.array([item[0]]), np.array([item[1]]), np.array([item[2]]), item[3], item[4]
        baseline_features = extract_baseline_features([s1s], [s2s])
        baseline_features = Variable(torch.from_numpy(baseline_features)).float()
        s1 = Variable(torch.from_numpy(s1))
        s2 = Variable(torch.from_numpy(s2))
        output = cnn(s1, s2, baseline_features)
        res.append(output.data.cpu().numpy()[0][0])

    # write prediction to file
    with open(args.predfile, 'w') as f:
        for i in res:
            f.write(str(i) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)