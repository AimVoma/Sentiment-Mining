###############################################################################
# Main utility class that performs ML Pipeline of self-trained embedding models
# Gensim Models: Word2Vec(S-Gram, CBOW), Doc2Vec(S-GRAM, CBOW)
# Glove
# -------------
# Corpus: Sentiment140
# Dimensions: [50]
###############################################################################
from emb_vector import Embedding_Vector,TFTransformer
from dataset_loader import *
from helper_functions import *
from sentiment_model import Sentiment_Model
from plot_helper import *
from tqdm import tqdm,trange
import pandas as pd
import random, sys, os, glob, re, pickle
import numpy as np
import glob
from sklearn.model_selection import train_test_split

class ML_pipeline:
    def __init__(self, mode=None, sample=None):

        self.emb_name = None
        self.mode = mode
        self.transformer = None
        self.sent_model = Sentiment_Model()
        self.info = {}
        self.models = []

        if self.mode == 'cross_domain':
            sample_ = ''

            try:
                loader_ = dataset_loader(self.mode)
            except KeyError:
                print('Dataset Not Found, loader_')
                sys.exit(-1)

            fit_corpus = loader_.load_devset()

            # Use the whole corpus or sample size
            tf_corpus, _, _, _ = train_test_split(
                        fit_corpus['x_data'][:sample], fit_corpus['y_labels'][:sample],
                        test_size=0.25, random_state=40,
                        shuffle=True)

            # If we perform sample to feed vectorizer on sentiment140 corpus
            if sample:
                sample_ = '_' + str(sample)
                tf_corpus = tf_corpus[:sample]

            self.transformer = TFTransformer()
            self.transformer.fit(tf_corpus)

            self.emb_name = 'tfidf[{0}]{1}'.format(self.mode, sample_)

        elif self.mode == 'tfidf':
            self.transformer = TFTransformer()
            self.emb_name = 'tfidf'
        elif self.mode == 'pre_trained' or self.mode == 'self_trained':
            self.transformer = Embedding_Vector()
        elif self.mode == 'contextual':
            pass
        else:
            raise NameError('Wrong mode Input!')

    def transform(self, models=['svm', 'lr', 'rf'], ensemble=False, sample=None):
        self.models = models
        for dataset_file in tqdm(glob.iglob('./pickled/datasets/*', recursive=True), total=4, desc='Dataset Processing: '):
            dataset_name = dataset_file.split('/')[-1]

            print('__________________________________')
            print('Loading Dataset: {} ......'.format(dataset_name))
            if '.pkl' not in dataset_name:
                try:
                    loader = dataset_loader(dataset_name)
                except KeyError:
                    print("{} does not exist as dataset".format(dataset_name))
                    sys.exit(-1)

            assert(loader is not None)
            # Only sample or whole Dset
            # if sample:
            #     print('Sample Used: ', str(sample))
            #     class_sample = devset['x_data'][:sample] + devset['x_data'][-sample:]
            #     class_lables = devset['y_labels'][:sample] + devset['y_labels'][-sample:]
            # else:
            #     print('Loading Entire Dev Set .......')
            #     class_sample = devset['x_data']
            #     class_lables = devset['y_labels']
            #     pass
            folder = self.select_folder(self.mode)

            for embedding_file in glob.iglob('./pickled/{}/*.pkl'.format(folder), recursive=True):
                devset = loader.load_devset()
                assert(len(devset['x_data']) == len(devset['y_labels']))

                emb_name = embedding_file.split('/')[-1].strip('.pkl').split('_')[-1]
                print('Loading Embeddings {} .......'.format(emb_name))

                print(emb_name)

                if emb_name == 'elmo' or emb_name == 'use':
                    pass
                else:
                    x_train, x_test, y_labels_train, y_labels_test = train_test_split(
                                devset['x_data'], devset['y_labels'],
                                test_size=0.25, random_state=40,
                                shuffle=True)

                    self.transformer.load_pickled(emb_name, pre_trained=True)
                    x_emb_train = self.transformer.transform(x_train)
                    x_emb_test = self.transformer.transform(x_test)

                    #Memory Release due to embeddings loading mem-capacity
                    self.transformer.release_memory()
                    # Check for float64 particles due to fault transformation
                    x_emb_train, y_labels_train = sanitize_embeddings(x_emb_train, y_labels_train)
                    x_emb_test, y_labels_test = sanitize_embeddings(x_emb_test, y_labels_test)


                # Create Embeddings Information Dictionary
                self.info['embedding'] = emb_name
                self.info['dataset'] = dataset_name

                if not ensemble:
                    self.sent_model.classify_batch(x_emb_train, y_labels_train,
                    x_emb_test, y_labels_test,
                    self.info,
                    self.transformer,
                    self.models)
                # else:
                #     self.sent_model.ensemble(x_emb_train, y_labels_train, x_emb_test, y_labels_test, self.info)


            self.info['embedding'] = self.emb_name
            self.info['dataset'] = dataset_name

            if not ensemble:
                self.sent_model.classify_batch(x_train, y_labels_train,
                                               x_test, y_labels_test,
                                               self.info,
                                               self.transformer,
                                               self.models)
    def select_folder(self, mode):
        try:
            return {
                  'self_trained' : 'trained_emb',
                  'pre_trained' : 'pretrained_emb',
                  'contextual' : 'contextual_pretrained',
                  }[mode]
        except KeyError:
            print('mode: {}, Not Found!'.format(mode))

    def train_vectors(self):
        snt140 = Sentiment140()
        snt140.load_dataset('sentiment140_cleaned')
        snt140.create_corpus(30000)

        corpus = snt140.get_corpus()

        embeddings = Embedding_Vector()
        embeddings.train('doc2vec', corpus, 50, 50, {'dm':1})
        embeddings.train('doc2vec', corpus, 50, 50, {'dm':0})

        embeddings.train('word2vec', corpus, 50, 50, {'sg': 1})
        embeddings.train('word2vec', corpus, 50, 50, {'sg': 0})
        embeddings.train('glove', corpus, dims=50, epochs=50)

    def plot_predictions(self, file='./pickled/models/*'):
        for model_file in glob.iglob(file, recursive=True):
            file = model_file.rsplit('\\')[1]
            full_path = './pickled/models/{}/*.pkl'.format(file)
            Plotter(dir=full_path).prepare_data(filters=['word2vec']).plot_bgraph(figure='{}_{}'.format('tf_idf',file))

    def __store_missclassified(self, df):
        predicted_ = list()
        for item in df['predicted']:
            predicted_.append(item.tolist())

        flags = [True for i in range(len(y_labels_test))]

        for prediction in predicted_:
            for index, element in enumerate(prediction):
                if element != y_labels_test[index]:
                    flags[index] = False
        counter=0
        for item in flags:
            if item == False:
                counter+=1

        filtered_list = list()
        for index, flag in enumerate(flags):
            if flag == False:
                filtered_list.append([ y_labels_test[index], X_test[index] ])

        df = pandas.DataFrame(filtered_list, columns=['sentiment', 'text'], index=None)

        df.to_csv('./missclassified.txt', header=None, index=None, sep='\t', mode='w')
