from collections import defaultdict
import pprint
import argparse
import difflib
from googletrans import Translator
import numpy as np
translator = Translator()
import time

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)
# parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--v', type=int, required=True)


def main(args):
    T0 =time.time()
    Sentence_embeddings = []
    with open(args.pairfile,'r') as f:
        sentences = []
        for line in f.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            if 2 == args.v: # translator is a little bit slow
                first_sentence = translator.translate(first_sentence, dest='en').text 
            sentences += [first_sentence, second_sentence]
    # sentence embeddings:
        for sent in sentences:
            sent_embedding = defaultdict(int)
            for w in sent.split():
                w = w.strip('?.,')
                sent_embedding[w] += 1
            Sentence_embeddings.append(sent_embedding)
        first_sents_embeddings = Sentence_embeddings[::2] 
        second_sents_embeddings = Sentence_embeddings[1::2] 

    with open(args.predfile,'w') as f:
        for i in range(len(first_sents_embeddings)):
            # cosine similarity
            Embedding1 = []
            Embedding2 = []
            for w in first_sents_embeddings[i]:
                Embedding1.append(first_sents_embeddings[i][w])
                Embedding2.append(second_sents_embeddings[i][w])
            if 0 == sum(Embedding2): ret = 0
            else: 
            #https://stackoverflow.com/questions/6709693/calculating-the-similarity-of-two-lists
            # https://docs.python.org/3/library/difflib.html
                sm= difflib.SequenceMatcher(None,Embedding1,Embedding2)
                ret = sm.ratio()*5 
            f.write(str(ret)+'\n')
    print("Elapsed time:",time.time() - T0)
        

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)