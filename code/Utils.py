from collections import Counter
from nltk import word_tokenize
import pdb

def get_vocab(f1, f2, f3):
    """get vocab from en-test, en-train and en-val, write to txt file"""
    vocab = Counter()
    with open(f1, 'r') as f_1, open(f2, 'r') as f_2, open(f3, 'r') as f_3:
        data1 = f_1.readlines()
        data2 = f_2.readlines()
        data3 = f_3.readlines()

    def get_sentence(data):
        """data: list of lines, each line: s1, s2, score"""
        res = []
        for line in data:
            line_sp = line.split('\t')
            res.extend(line_sp[:2])
        return res

    sentences = []
    sentences.extend(get_sentence(data1))
    sentences.extend(get_sentence(data2))
    sentences.extend(get_sentence(data3))

    for sentence in sentences:
        words = word_tokenize(sentence.decode('utf-8'))
        for word in words:
            vocab[word] += 1
    sort_vocab = vocab.most_common()
    with open("./data/vocab.txt", "w") as f:
        for item in sort_vocab:
            f.write("{}\t{}\n".format(item[0].encode('utf-8'), item[1]))

def to_lower(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = [line.lower() for line in lines]
    with open(fname, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    # to_lower("./data/en-train.txt")
    # to_lower("./data/en-test.txt")
    # to_lower("./data/en-val.txt")
    get_vocab("./data/en-train.txt", "./data/en-test.txt", "./data/en-val.txt")
