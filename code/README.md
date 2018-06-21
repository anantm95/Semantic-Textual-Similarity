## Published Baseline Reimplementation

**[DT_Team at SemEval-2017 Task 1: Semantic Similarity Using Alignments, Sentence-Level Embeddings and Gaussian Mixture Model Output](http://www.aclweb.org/anthology/S17-2014)**

To get the predicted score of development data: 

`python3 baseline.py --pairfile ../data/en-train.txt --valfile ../data/en-val.txt --predfile ../data/pred-en-val.txt --v 1`

To get the predicted score of test data: (Train on both training data and dev data)

`python3 baseline.py --pairfile ../data/all-train-dev.txt --valfile ../data/en-test.txt --predfile ../data/pred-en-test.txt --v 1`

Evaluation: (compare the pearson correlation between predicted scores and gold labels)

`python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en-val.txt`

`python3 evaluate.py --goldfile ../data/en-test.txt --predfile ../data/pred-en-test.txt`

## Extensions
To get the predicted score, simply change the .py with corresponding extention_XXX.py file (extension one is included in baseline.py code)

e.g.
`python3 extension_cnn.py --pairfile ../data/all-data-shuffled.txt --valfile ../data/en-test.txt --predfile ../data/pred-en-lstm-test.txt --v 1`

The evaluation command is the same as above.

