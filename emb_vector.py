###############################################################################
# Main utility class(Emb. Vector) for creating word vector models
# [pretrained/self-trained], W2V, Doc2Vec, Glove, FastText, Elmo
# Custom TFIDF-Transformer Class
###############################################################################
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# _____________________________________________________________
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np
import pickle

import cython
# _____________________________________________________________
import tf_glove
import tensorflow_hub as hub
import tensorflow as tf
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec,KeyedVectors
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import sys
import gc
# from pytorch_fast_elmo import FastElmo, batch_to_char_ids
import inspect

##########################################################
# Custom TFIDF Class for Sklearn Pipeline(Tuning)
##########################################################
class TFTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, min_df=1, ngram_range=(1,1), norm=None, max_features=None):
        self.vectorizer = None

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def transform(self, sentences, y=None):
        return self.vectorizer.transform(sentences)

    def fit(self, corpus, y=None):
        self.vectorizer = TfidfVectorizer(token_pattern=r'\w{2,}',
                                          sublinear_tf=True,
                                          analyzer='word',
                                          ngram_range=self.ngram_range,
                                          min_df=self.min_df,
                                          max_features = self.max_features,
                                          norm=self.norm)
        self.vectorizer.fit(corpus)
        return self

    def fit_transform(self, corpus, y=None):
        self.vectorizer = TfidfVectorizer(token_pattern=r'\w{2,}',
                                          sublinear_tf=True,
                                          analyzer='word',
                                          ngram_range=self.ngram_range,
                                          min_df=self.min_df,
                                          norm=self.norm)
        self.vectorizer.fit(corpus)
        return self.transform(corpus)

    def get_feature_names(self):
        try:
            return self.vectorizer.get_feature_names()
        except:
            return False

##########################################################
# Main Word Vectors Class For Training/Loading/Transforming
# words to word vectors
##########################################################
class Embedding_Vector:
    def __init__(self):
        self.embeddings = None
        self.dim = None
        self.count_vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer(norm='l2')
        self.word2tfidf = None
        self.embedding_type = 'word_vectors'

    def train(self, emb_type, corpus, dims=50, epochs=50, *args):
        documents = list(map(lambda document: document.split(), corpus))
        if emb_type is 'doc2vec':
            print("Training Doc2Vec Dimensions[{0}], epochs[{1}]".format(str(dims), str(epochs)))

            if args:
                for arg in args:
                    try:
                        dm = int(arg.get('dm'))
                    except KeyError:
                        pass

            model = Doc2Vec(size=dims, window=5, min_count=5,negative=5, workers=4, dm=dm, alpha=0.025)

            tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
            model.build_vocab(tagged_data)

            model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)

        elif emb_type is 'word2vec':
            print("Training Word2Vec Dimensions[{0}], epochs[{1}]".format(str(dims), str(epochs)))

            if args:
                for arg in args:
                    try:
                        sg = int(arg.get('sg'))
                    except KeyError:
                        pass

            model = Word2Vec(documents, size=dims, window=5, min_count=5, workers=4, alpha=0.025, sg=sg)
            model.train(documents, total_examples=len(documents), epochs=epochs)
        else:
            print("Training Glove Dimensions[{0}], epochs[{1}]".format(str(dims), str(epochs)))

            glove_model = tf_glove.GloVeModel(embedding_size=dims, context_size=5, min_occurrences=5,learning_rate=0.05, batch_size=512)
            glove_model.fit_to_corpus(documents)
            model = glove_model.train(num_epochs=epochs)

        self.embeddings = model
        if args and emb_type=='doc2vec':
            dm = 'DM' if dm == 1 else 'DBOW'
            self.save_pickled("{0}{1}_{2}".format(emb_type, str(dims), dm), pre_trained=False)
        if args and emb_type=='word2vec':
            sg = 'SG' if sg == 1 else 'CBOW'
            self.save_pickled("{0}{1}_{2}".format(emb_type, str(dims), sg), pre_trained=False)
        else:
            self.save_pickled("{0}{1}".format(emb_type, str(dims)), pre_trained=False)
        print("Trained Model saved!")

    # Dummy Instance for sklearn Pipeline
    def fit(self, X_DUMMY_TRAIN, Y_DUMMY_TRAIN):
        return self

    def transform(self, text):
        if self.embedding_type == 'elmo' or self.embedding_type == 'use':
            if(self.embedding_type == 'elmo'):
                self.embeddings = self.embeddings(text,signature="default",as_dict=True)["elmo"]
            else:
                self.embeddings = self.embeddings(text)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            raw_sentence_embeddings = sess.run(self.embeddings)

            final_sentence_embeddings = []
            for sentence_embedding in raw_sentence_embeddings:
                final_sentence_embeddings.append(np.ndarray.mean(sentence_embedding, axis=0))

            return final_sentence_embeddings

        elif self.embedding_type =='doc2vec':
            processed_text = (list(map(lambda sentence: simple_preprocess(sentence), text)))
            return(list(map(lambda emb: self.embeddings.infer_vector(emb), processed_text)))
        else:
            # if isinstance(self.embeddings, tf_glove.glove_vector):
            if isinstance(self.embeddings, int):
                # GLOVE SELF TRAINED______(INSTANCE INT DUMMY STATEMENT)
                return np.array([
                    np.mean([self.embeddings.get_emb(word) for word in words.split()], axis=0)
                    for words in text])
            else:
                # GENERAL EMBEDDINGS(W2V, GLOVE, FTEXT)______
                return np.array([
                    np.mean([self.embeddings[word] for word in words.split() if word in self.embeddings.wv]
                            or [np.zeros(self.dim)], axis=0)
                    for words in text])

    def transform_tf(self, text, init_vocab=False):
        if not init_vocab:
            return np.array([
                np.mean([self.embeddings[word] * self.word2tfidf.get(word, 1) for word in words.split() if word in self.embeddings.wv]
                        or [np.zeros(self.dim)], axis=0)
                for words in text
            ])
        else:
            # convert text data into term-frequency matrix
            count_data = self.count_vectorizer.fit_transform(text)

            # convert term-frequency matrix into tf-idf
            self.tfidf_transformer.fit(count_data)

            # create dictionary to find a tfidf word each word
            self.word2tfidf = dict(zip(self.count_vectorizer.get_feature_names(), self.tfidf_transformer.idf_))

    def load_trained(self, embeddings, dimensions, training_model):
        if embeddings == 'doc2vec':
            self.embeddings = Doc2Vec.load('./pickled/trained_emb/{0}{1}_{2}.pkl'.format(embeddings, str(dimensions), training_model))
            self.embedding_type = embeddings
        if embeddings == 'word2vec':
            self.embeddings = Word2Vec.load('./pickled/trained_emb/{0}{1}_{2}.pkl'.format(embeddings,str(dimensions), training_model))
            self.embedding_type = embeddings
        if embeddings == 'glove':
            # self.embeddings = Doc2Vec.load('./pickled/trained_emb/d2v_100.pkl')
            # self.embedding_type = embeddings
            pass

    def load_pretrained(self, embeddings_file, save_pickled=False):

        try:
            embeddings = embeddings_file.split('\\')[1].rsplit('_')[0]
            dimensions = embeddings_file.split('\\')[1].rsplit('_')[1]
        except IndexError:
            embeddings = embeddings_file
            pass

        print('________Loading Pretrained Vectors_______')
        if embeddings == 'word2vec':
            print("w2v_{0}".format(str(dimensions)))
            try:
                self.embeddings = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
            except UnicodeDecodeError as error:
                self.embeddings = KeyedVectors.load_word2vec_format(embeddings_file, binary=True)

        elif embeddings == 'glove':
            print("glove_{0}".format(str(dimensions)))
            tmp_file = get_tmpfile("glove_word2vec.txt")
            glove2word2vec(embeddings_file, tmp_file)
            self.embeddings = KeyedVectors.load_word2vec_format(tmp_file, binary=False, unicode_errors='ignore')

        elif embeddings == 'elmo':
            print("elmo_{0}".format("1024"))

            self.embeddings = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
            self.embedding_type = 'elmo'
            dimensions=1024

        elif embeddings == 'use':
            print("use_{0}".format("512"))

            self.embeddings = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/3")
            self.embedding_type = 'use'
            dimensions=512

        elif embeddings == 'ftext':
            print("ftext_{0}".format(str(dimensions)))

            self.embeddings = KeyedVectors.load_word2vec_format(embeddings_file)
        else:
            raise NameError('Wrong Embedding File')

        if save_pickled:
            print('Saving as pickled .... {0}[{1}]'.format(embeddings, dimensions))
            self.save_pickled('{0}[{1}]'.format(embeddings, dimensions), pre_trained=True)

        return self
    def load_pickled(self, name, pre_trained=True):
        if not pre_trained:
            self.embeddings = joblib.load("./pickled/trained_emb/{}.pkl".format(name))
        else:
            self.embeddings = joblib.load("./pickled/pretrained_emb/{}.pkl".format(name))

    def load_contextual_devset(self, name):
        return joblib.load("./pickled/contextual_pretrained/{}.pkl".format(name))

    def save_pickled(self, name, pre_trained=True):
        if not pre_trained:
            joblib.dump("./pickled/trained_emb/{}.pkl".format(name), self.embeddings)
        else:
            joblib.dump("./pickled/pretrained_emb/{}.pkl".format(name), self.embeddings)

    def release_memory(self):
        print('Releasing Memory: ')
        self.embeddings = None
        # gc.collect()

##########################################################
# OOP Approach for Embedding Class(Not Implemented)
##########################################################

# class BaseEmbeddingVector:
#     subclasses = {}
#     def __init__(self, embeddings=None, dims=None):
#         self.embeddings = embeddings
#         self.dims = dims
#
#     @abstractmethod
#     def embed(self, *args, **kwargs):
#         raise NotImplementedError
#
#     @abstractmethod
#     def load_embeddings(self, *args, **kwargs):
#         raise NotImplementedError
#
#     @classmethod
#     def _find_file_format(cls, file_name):
#         pass
#
#     @classmethod
#     def create(cls, file_name, **kwargs):
#         parsed = cls._emb_file_parse(file_name)
#         if parsed['emb'] not in cls.subclasses:
#             raise ValueError("Bad parsed embedding type '{}'".format(parsed['emb']))
#         return cls.subclasses[parsed['emb']].load_embeddings(file_name, b_format=kwargs.get('b_format', cls._find_file_format(file_name)))
#
#     @classmethod
#     def _emb_file_parse(cls, embedding):
#         return dict(zip(['emb', 'dim', 'domain'], embedding.split('_')))
#
#     @classmethod
#     def print_info(cls, dimensions, domain):
#         print('________Loading Pretrained ' + cls.__name__ + ' Vectors_______')
#         print(str(dimensions) + '_' + domain)
#
#     @classmethod
#     def register_subclass(cls, embeddings_type):
#         def decorator(subclass):
#             cls.subclasses[embeddings_type] = subclass
#             return subclass
#         return decorator
#
#
# class WordEmbeddings(BaseEmbeddingVector):
#     def embed(self, *args, **kwargs):
#         # print(args[0])
#         # print(self.embeddings)
#         # print(self.dims)
#
#
#         return np.array([np.mean([self.embeddings[w] for w in words.split() if w in self.embeddings.wv] or [np.zeros(self.dims)], axis=0) for words in args[0]])
#
#
# class ContextEmbeddings(BaseEmbeddingVector):
#     def embed(self, *args, **kwargs):
#         init = tf.initialize_all_variables()
#         sess = tf.Session()
#         sess.run(init)
#         embedding_val = sess.run(self.embeddings)
#
#         final_emb = []
#         for emb in embedding_val:
#             final_emb.append(ndarray.mean(emb, axis=0))
#         return final_emb
#
#
# @BaseEmbeddingVector.register_subclass('w2v')
# class Word2Vec(WordEmbeddings):
#
#     @classmethod
#     def load_embeddings(cls, embeddings_file, b_format=False):
#         _temp_dict = cls._emb_file_parse(embeddings_file)
#         cls.print_info(_temp_dict['dim'], _temp_dict['domain'])
#         w2v_file = './Embeddings/' + _temp_dict['emb'] + '_' + _temp_dict['dim'] + '_' + _temp_dict['domain']
#         # cls.embeddings = KeyedVectors.load_word2vec_format(w2v_file, binary=b_format)
#         return Word2Vec(embeddings=KeyedVectors.load_word2vec_format(w2v_file, binary=b_format), dims=int(_temp_dict['dim']))
#         # return KeyedVectors.load_word2vec_format(w2v_file, binary=b_format)
#         # return Word2VecGG()
#         # return cls
