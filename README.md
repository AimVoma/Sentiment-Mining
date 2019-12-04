# Sentiment-Mining
Graduation Project - Empirical Survey On Text Representations In Sentiment Analysis

###### Project Description 
This is a part of a graduation project which included among others Natural Language Processing(NLP), Information Retrieval(IR) and Sentiment Analysis as main experimentational objectives.

This project was developed on Atom(editor) with Hydrogen plug-in for prototyping and early development reasons. The main execution of project's functionality is in main.py. Fine-Tuning elmo is served as standalone execution file(fine_tune.py). The rest of the Python modules are served as basic classes for supported functionality of the project.

* Support Python Modules
 * Load/Save Datasets Module(dataset_loader.py)
 * Word Vectors Class Implementation(emb_vector.py)
 * Sentiment Analysis Models Sklearn(LR, MNB, Adaboost(RF), SVM), Ensemble (sentiment_model.py)
 * Glove Algorithm - Python Implementation MIT(glove.py)
 * NLP pre-processing, T-SNE(k-closest), Sanitize word vectors(helper_functions.py)
 * Machine Learning(ML) Pipeline of training word vectors on Sentiment140(ml_pipeline.py)
 * Scatter Plot, H-Bar Plot, Correlation Plot(plot_helper.py)

###### Project Functionality
* Rule-based text preprocessing and normalization with SpaCy, Gensim, NLTK
* Sparse Vector Space Models Experiments
  * Cross-Domain experiments with TFIDF and sentiment datasets(ImDb, Sentiment140, Sarcasm)
  * Eli5 Sparse model(SVM n-gram) analysis in terms of false positive/missclasified sentiment data
* Word vector models(Doc2Vec, Word2vec - S-gram, CBOW) & Glove analysis on sentiment corpora(Sentiment140)
  * Identifying sentiment outliers in Sentiment Orientation(SO) analysis with T-SNE of the produced embeddings
* Pre-trained word and contextual Language Models(LM) performance on sentiment datasets(ImDb, Sentiment140, SemEval_2016)
* Fine-Tuning pre-trained Elmo model weights on unseen sentiment data with Keras(Embedding Layer)


###### Technical
* Gensim, SpaCy, NLTK, scikit-learn, Tensorflow, Keras

###### Prerequisite: [allennlp](https://github.com/allenai/allennlp) NLP Library
