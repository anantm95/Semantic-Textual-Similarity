
# Semantic Textual Similarity

#### Anant Maheshwari&emsp;&emsp;&emsp;&emsp;&emsp;Simeng Sun&emsp;&emsp;&emsp;&emsp;&emsp;Danni Ma&emsp;&emsp;&emsp;&emsp;&emsp;Yezheng Li


## Abstract
Semantic Textual Similarity (STS) measures the meaning similarity of sentences. Applications of this task include machine translation, summarization, text generation, question answering, short answer grading, semantic search, dialogue and conversational systems. We developed Support Vector Regression model with various features including the similarity scores calculated using alignment-based methods and semantic composition based methods. We have also trained sentence semantic representations with BiLSTM and Convolutional Neural Networks (CNN). The correlations between our system output the human ratings were above 0.8 in the test dataset.

## Introduction
The goal of this task is to measure semantic textual similarity between a given pair of sentences (what they mean rather than whether they look similar syntactically). While making such an assessment is trivial for humans, constructing algorithms and computational models that mimic human level performance represents a difficult and deep natural language
understanding (NLU) problem.

#### Example 1:

English: Birdie is washing itself in the water basin.

English Paraphrase: The bird is bathing in the sink.

Similarity Score: 5 ( The two sentences are completely equivalent, as they mean the same thing.)

#### Example 2:

English: The young lady enjoys listening to the guitar.

English Paraphrase: The woman is playing the violin.

Similarity Score: 1 ( The two sentences are not equivalent, but are on the same topic. )

Semantic Textual Similarity (STS) measures the degree of equivalence in the underlying semantics of paired snippets of text. STS differs from both textual entailment and paraphrase detection in that it captures gradations of meaning overlap rather than making binary classifications of particular relationships. While semantic relatedness expresses a graded semantic relationship as well, it is non-specific about the nature of the relationship with contradictory material still being a candidate for a high score (e.g., “night” and “day” are highly related but not particularly similar). The task involves producing real-valued similarity scores for sentence pairs. Performance is measured by the Pearson correlation of machine scores with human judgments.

## Experimental Design

### Data
We obtained the data by merging data from year 2012 to 2017 SemEval Shared Task. Out of a total of approximately 28000 sentence pairs, we were left with about 15115 sentence pairs (cleaning involved removing sentence pairs without a tab delimiter and pairs with a blank gold score). We split the data into three parts as below. 

Training data: 13365 pairs
Validation data:  1500 pairs
Test data: 250 pairs (Same as used by the other teams to test their model in the 2017 task)

#### Data Pre-processing

We used tokenization and lemmatization on the data as a pre-processing step before we turn the data into our models. We chose to do this step as lemmatization does not take away any semantic information from sentences and hence was an essential step for our application.

### Evaluation Metric
The official score is based on weighted Pearson correlation between predicted similarity and human annotated similarity. The higher the score, the better the the similarity prediction result from the algorithm.

### Simple baseline

For the simple baseline, we used an unsupervised approach by creating sentence vectors with each dimension representing whether an individual word appears in a sentence. The final score is calculated using cosine similairty between the sentence vectors. 

We achieved the following results using the simple baseline: 

| Model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| Simple baseline     | 0.428           | 0.633     |

## Experimental Results

### Published baseline

We re-implemented DT\_Team's work[2], as this is the state-of-the-art for monolingual STS task. DT\_Team used linear Support Vector Regression with a set of 7 features. Among them, two were only compatible with Windows, and one feature did not improve the performance at all, so we re-implemented four of them. Details are below.

1. **Unigram Overlap**: Count the unigram overlap between sentence A and sentence B, with synonym check. Then Calculate the ration of overlap count over the total length of two sentences.
2. **Word Alignment**: Look for each word of sentence A in Wordnet, we can get a synonym set (synset). We align this word with a corresponding one in sentence B by using path similarity in Wordnet trees. The labeled alignments are used as one useful feature.
3. **Absolute Difference**: Let Cta and Cta be the counts of tokens of type t ∈ {all tokens, adjectives, adverbs, nouns, and verbs} in sentence A and B respectively. We calculate the absolute difference as |Cta−Ctb|/(Cta+Ctb).
4. **Min to Max Ratio**: Let Ct1 and Ct2 be the counts of type t ∈ {all, adjectives, adverbs, nouns, and verbs} for shorter Sentence 1 and longer Sentence 2 respectively. We calculate the minimum to maximum ratio Ct1/Ct2 as one feature.

The published performance of DT\_Team model on the same test data is 0.8536. However, we could only achieve 0.6989 after re-implementation. One reason is that there are two features which might be useful, but we did not use due to operating system compatibility. It is also possible that DT\_Team had some pre-processed steps which they did not illustrate very specifilly in the paper, and we did not fully implement them. However, we focus on our extensions in an attempt to reach close to the state of the art performance.

<center>

| Model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| Simple baseline     | 0.428           | 0.633     |
| Published baseline  | 0.6114          | 0.6989    |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

### Extensions

1. Resnik Similarity using Information Content from Brown Corpus

We used information content generated from the Brown corpus to compute the resnik similarity between paths in the wordnet trees for the given sentence pairs. This approach uses IC of the Least Common Subsumer (most specific ancestor node) to output a score which is used by the Support Vector Regression model. 

We were able to improve upon our model by a slight amount using this extension:
<center>

| Model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + IC       | 0.6226          | 0.7097    |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

2. Convolution Neural Networks to generate sentence embeddings

![alt text](https://raw.githubusercontent.com/SimengSun/CIS530-project/master/deliverables/images/cnn-model.png "cnn-model")

The 2nd extension we implemented is a Convolutional neural network, which produces dense representation of sentences as by-product. We have two version of CNN model for this extension.

The first version is shown above. We stack the two sentence together by column and form a matrix of size N by 2D, D is the dimension of word embedding, N is the maximal length of sentence the model can take in. We use pretrained GoogleNews vector, D equals to 300. When the length of sentence is not long enough, we employ masks to cover the empty space. After convolution operation with kernel sizes range from 2 to 5, each with 16 output channels, a dense vector that encodes the semantic relationship of two sentences is generated. We then combine our baseline features and feed this vector in a linear regression model to predict similarity score. 

The second version is shown below. Unlike the first version, we use CNN to encode each sentence separately and concatenate two sentence embedding and baseline features, then feed this new vector into a linear regression model. This model performs better than the last one, the best result we got is shown in the table below.

![alt text](https://raw.githubusercontent.com/SimengSun/CIS530-project/master/deliverables/images/cnn-model-1.png "cnn-model-1")

<center>

| Model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + CNN      | 0.6615          | 0.6460    |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

3. Use InferSent trained sentence embeddings

[InferSent](https://research.fb.com/downloads/infersent/) is a sentence embeddings method that provides semantic sentence representations. It is trained on natural language inference data and generalizes well to many different tasks.

We use InferSent to get the embeddings of all the sentences we have. Given a pair of sentences, if they are semantically similar, the cosine similary between two sentence embeddings are supposed to be high. We extracted the cosine similarity between sentence pairs, added it as a feature, and fed to our Support Vector Regression model.

With the help of InferSent trained sentence representations, the model outperforms baseline model on both validation data and test data:
<center>

| Model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + InferSent|**0.7220**       |**0.8104** |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

### Error Analysis

Altough Pearson correlation is widely used as evaluation criteria in the literature, it is not quite intepretable. We introduce two extra error analysis:
1. "Ave 5": average over predicted similarities of pairs with gold standard 5.0. The closer "Ave 5" to 5.0, the better the model.
2. "Ave 0": average over predicted similarities of pairs with gold standard 0.0. closer "Ave 5" to 0.0, the better the model.


|                    | Validation Set |            |            | Test Set |           |           |
|--------------------|----------------|------------|------------|----------|-----------|-----------|
|                    | Pearson         | Ave 5(128) | Ave 0(131) | Pearson  | Ave 5(10) | Ave 0(19) |
| Simple Baseline    | 0.428          | 3.274      | 0.532      | 0.633    | 4.088     | 0.623     |
| Published Baseline | 0.611          | 3.994      | 0.668      | 0.698    | 4.347     | 0.861     |
| CNN                | 0.661          | 4.030      | 0.569      | 0.646    | 3.767     | 0.561     |
| LSTM               | 0.722          | 4.076      | 0.258      | 0.810    | 4.427     | 0.516     |
| Gold Standard      | 1              | 5          | 0          | 1        | 5         | 0         |

Some observations:
1. We can see the trend is that (generally) "Ave 5" and "Ave 0" are improved as Pearson correlation is improved; level of improvement may vary: take (Validation Set of CNN and LSTM) as an example, "Ave 0" is improved from 0.569 to 0.258 while "Ave 5" is only improved from 4.030 to 4.076.
2. Since Test Set has only 250 pairs compaired to Validation Set (1500 pairs), we get only 10 pairs with gold standard 5 and only 19 pairs with gold standard 0 -- Test Set's result varies a lot -- take (Test Set of Published and CNN) as an example, Pearson correlation and "Ave 5" drop while "Ave 0" is improved. To conclude, "Ave 5" and "Ave 0" are less convincing since Test Set is not big.

## Conclusions

The state-of-art performs for this particular task is a test score of 0.85 on the test set for 2017. With our baseline features and best performing extension, we were able to reach a score of 0.8104 on the test set. We tried many approaches involving binary bag-of-words, word2vec embeddings, fastext embeddings for en-es cross-lingual pairs and sent2vec embeddings. Binary bag-of-words, although being a very simple model tends to perform very well in general and is a high-performing baseline. We also tried lots of different features involving path-similarity and resnik similarity in wordnet trees for the given sentences. One would hope that CNN's would perform well by capturing semantic information in the generated embeddings but are still unable to beat the best score we obtained. However as shown in the error analysis, CNN does have a good potential on this application and would certainly outperform other models if more data is available. 

## References

[1] Cer et. al, **[SemEval-2017 Task 1: Semantic Textual Similarity
Multilingual and Cross-lingual Focused Evaluation.](https://www.aclweb.org/anthology/S/S17/S17-2001.pdf)** *In Proceedings of the 11th International Workshop on Semantic Evaluations (SemEval-2017)*

[2] Maharjan et. al, **[DT Team at SemEval-2017 Task 1: Semantic Similarity Using Alignments, Sentence-Level Embeddings and Gaussian Mixture Model Output.](http://www.aclweb.org/anthology/S17-2014)** *In Proceedings of the 11th International Workshop on Semantic Evaluations (SemEval-2017)*

[3] Banjade et. al, **[DTSim at SemEval-2016 Task 1: Semantic Similarity Model Including Multi-Level Alignment and Vector-Based Compositional Semantics.](http://www.aclweb.org/anthology/S16-1097)** *In Proceedings of SemEval-2016*

[4] Conneau et. al, **[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](http://aclweb.org/anthology/D17-1070)** *In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)*

## Acknowledgements

We took part in regular meetings with out mentor TA Nitish Gupta who helped us with his thoughts on our ideas and giving us possible directions for our extensions to improve our results. 
