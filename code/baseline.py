from collections import defaultdict
import pprint
import argparse
import difflib
from googletrans import Translator
import numpy as np
import pickle
translator = Translator()
import time
from nltk.corpus import wordnet_ic
from gensim.models import KeyedVectors
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
# import pdb

from nltk.corpus import wordnet as wn
brown_ic = wordnet_ic.ic('ic-brown.dat')
vec1 = KeyedVectors.load_word2vec_format("../data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
vec2 = KeyedVectors.load_word2vec_format("../data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)
parser.add_argument('--valfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--v', type=int, required=True)

# def get_ppdb(fname):
#     with open(fname, 'rb') as f:
#         ppdb = pickle.load(f)
#     return ppdb

# # PPDB_file = 'ppdb-filter'
# PPDB_file = 'ppdb-2.0-l-lexical'
# ppdb = get_ppdb('../' + PPDB_file)
# print('end loading ' + PPDB_file)

#--------------------
def sentence_similarity_simple_baseline(s1, s2,counts = None):
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
            Embedding1.append(first_sent_embedding[w] * 1.0/ (counts[w]+0.001))
            Embedding2.append(second_sent_embedding[w] *1.0/ (counts[w]+0.001))
    else:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w])
            Embedding2.append(second_sent_embedding[w])
    ret_score = 0
    if not 0 == sum(Embedding2): 
        #https://stackoverflow.com/questions/6709693/calculating-the-similarity-of-two-lists
        # https://docs.python.org/3/library/difflib.html
        sm= difflib.SequenceMatcher(None,Embedding1,Embedding2)
        ret_score = sm.ratio()*5 
    return ret_score

#--------------------
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

        # compute ppdb score
        # if len(L) > 0 and max(L) > 0.6:
        #     align_cnt += 1
        #     ss2 = synsets2[L_prime.index(max(L))]
        #     w1, w2 = synset.lemma_names()[0], ss2.lemma_names()[0]
        #     if (w1, w2) in ppdb:
        #         ppdb_score += ppdb[(w1, w2)]
        #     elif (w2, w1) in ppdb:
        #         ppdb_score += ppdb[(w2, w1)]
        #     else:
        #         if w2 == w1:
        #             ppdb_score += 5
        #         else:
        #             ppdb_score += 0
        #         notin_cnt[0] += 1


        # Check that the similarity could have been computed

        if L: 
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count >0: score /= count

    # ppdb_wa_pen_ua features
    # len_s1, len_s2 = len(sentence1), len(sentence2)
    # ppdb_score = (1 - 0.4 * (len_s1 + len_s2 - 2*align_cnt) / float(len_s1 + len_s2)) * ppdb_score
    return score#, ppdb_score
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
    if(len(s1) > len(s2)):  shorter = 2

    s1, s2 = word_tokenize(s1), word_tokenize(s2)
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
    if count >0: score /= count
    return score

def extract_res_vec_similarity(s1, s2):
    first_sents_embeddings = np.empty([0,300])
    second_sents_embeddings = np.empty([0,300])

    first_vecs = np.array([])
    for w in s1.split():
        w = w.strip('?.,')
        if w in vec1:
            first_vec = np.array([vec1[w]])
            if first_vecs.shape[0] == 0:
                first_vecs = first_vec
            else:
                first_vecs = np.vstack((first_vecs, first_vec))
        else:
            if first_vecs.shape[0] == 0:
                first_vecs = np.random.normal(0, 5, 300)
            else:
                first_vecs = np.vstack((first_vecs, np.random.normal(0, 5, 300)))
        # print("first ")
        # print(first_vecs.shape)
    if(first_vecs.shape == (300, )):
        temp = first_vecs
    else:
        temp = np.mean(first_vecs, axis=0)
    # print(temp.shape)
    first_sents_embeddings = np.append(first_sents_embeddings, [temp], axis=0)

    second_vecs = np.array([])  
    for w in s2.split():
        w = w.strip('?.,')
        if w in vec2:
            second_vec = np.array([vec2[w]])
            if second_vecs.shape[0] == 0:
                second_vecs = second_vec
            else:
                second_vecs = np.vstack((second_vecs, second_vec))
        else:
            if second_vecs.shape[0] == 0:
                second_vecs = np.random.normal(0, 5, 300)
            else:
                second_vecs = np.vstack((second_vecs, np.random.normal(0, 5, 300)))
        # print("second ")
        # print(second_vecs.shape)
    if(second_vecs.shape == (300,)):
        temp = second_vecs
    else:
        temp = np.mean(second_vecs, axis=0)
    # print(temp.shape)
    second_sents_embeddings = np.append(second_sents_embeddings, [temp], axis=0)

    for i in range(len(first_sents_embeddings)):
        # cosine similarity

        ret = np.dot(first_sents_embeddings[i], second_sents_embeddings[i]) / (np.linalg.norm(first_sents_embeddings[i]) * np.linalg.norm(second_sents_embeddings[i]))
        ret = 5*(ret + 1) / 2

    return ret


# ==================     ======================
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

import gensim
import os
import collections
import smart_open
import random

def extract_doc2vec_similarity(s1,s2, model):
    s1 = [w.strip('?.,') for w in s1.split()]
    s2 = [w.strip('?.,') for w in s2.split()]
    embed1 = model.infer_vector(s1)
    embed2 = model.infer_vector(s2)
    ret = np.dot(embed1,embed2)
    return ret
#--------------------
# from sklearn.svm import SVC, 
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression
import sklearn
def main(args):
    T0 =time.time()
    #----------
    # training 
    first_sents = []
    second_sents = []
    true_score = []

    with open(args.pairfile,'r') as f:
        for line in f.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            gs = line_split[2]
            if 2 == args.v: # translator is a little bit slow
                first_sentence = translator.translate(first_sentence, dest='en').text 
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)
            true_score.append(gs)

    Counts_for_tf = defaultdict(int)

    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1

    # def read_corpus(fname, tokens_only=False):
    #     with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    #         for line in f:
    #             line_split = line.split('\t')
    #             for sent in line_split:
    #                 if tokens_only:
    #                     yield gensim.utils.simple_preprocess(sent)
    #                 else:
    #                     # For training data, add tags
    #                     yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(sent), [0])

    # train_corpus = list(read_corpus(args.pairfile))
    # model_doc2vec = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=55)
    # model_doc2vec.build_vocab(train_corpus)
    # print("========== doc2vec model training start ========")
    # T1 = time.time()
    # model_doc2vec.train(train_corpus, total_examples=model_doc2vec.corpus_count, epochs=model_doc2vec.epochs)
    # print("Elapsed time:", time.time() - T1)
    # print("========== doc2vec model training finished =====")
    feature_scores = []
    N = len(first_sents)
    T1 = time.time()
    for i in range(N):
        s1 = first_sents[i]
        s2 = second_sents[i]


        scores = [ sentence_similarity_simple_baseline(s1,s2, Counts_for_tf),
                   sentence_similarity_word_alignment(s1,s2)
                   ,sentence_similarity_information_content(s1,s2)
                   , extract_overlap_pen(s1, s2)
                   ,*extract_absolute_difference(s1, s2)
                   ,*extract_mmr_t(s1, s2)
                   #,extract_res_vec_similarity(s1, s2)
                 ]
                   #, extract_doc2vec_similarity(s1,s2, model_doc2vec)]
        # cosine similarity
        feature_scores.append(scores)
        if 0 == (i+1) % int(N/10): print("%.2f" % ( (i +1)*1.0/ N *100), "%"+"finished (",time.time() - T1,")")

    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(feature_scores); X_features = scaler.transform(feature_scores)
    print("Elapsed time:",time.time() - T0,"(preprocessing)")
    #clf = LinearRegression(); clf.fit(X_features, true_score)
    # clf = BaggingRegressor(SVR(kernel='linear'), n_estimators=15) # R1 uses default parameters as described in SVR documentation
    clf = SVR(kernel='linear')
    clf.fit(X_features, true_score)


    # cross validation on SVR parameters here
    # for c in [0.001, 0.01, 0.1, 1, 10, 100]:
    #     clf = SVR(kernel='linear', C=c)
    #     res = cross_val_score(clf, X_features, true_score)
    #     print (sum(res) / len(res))

    #-----------
    # predicting
    first_sents = []
    second_sents = []
    with open(args.valfile,'r') as f_val:
        for line in f_val.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            if 2 == args.v: # translator is a little bit slow
                first_sentence = translator.translate(first_sentence, dest='en').text 
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)


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

        scores = [ sentence_similarity_simple_baseline(s1,s2, Counts_for_tf),
                   sentence_similarity_word_alignment(s1,s2)
                   ,sentence_similarity_information_content(s1,s2)
                   ,extract_overlap_pen(s1, s2)
                   ,*extract_absolute_difference(s1, s2)
                   ,*extract_mmr_t(s1, s2)
                   #,extract_res_vec_similarity(s1, s2)
                   ]
                   #, extract_doc2vec_similarity(s1,s2, model_doc2vec) ]
        # cosine similarity
        feature_scores.append(scores)
        if 0 == (i+1) % int(N/10): print("%.2f" % ((i+1) *1.0 / N *100),"%"+"finished (",time.time() - T1,")")
    X_features = scaler.transform(feature_scores)
    Y_pred_np = clf.predict(X_features)
    Y_pred_np = [min(5,max(0,p),p) for p in Y_pred_np]
    with open(args.predfile,'w') as f_pred:
        for i in range(len(Y_pred_np)):
            f_pred.write(str(Y_pred_np[i])+'\n')
    print("Elapsed time:",time.time() - T0)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
