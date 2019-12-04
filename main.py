##################################################################
# Main Module for run-time experiment execution with Hydrogen(Atom)
##################################################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import random, sys, os, glob, re, pickle
import numpy as np
import glob
import re
import pandas as pd

# Custom Module Imports
from emb_vector import Embedding_Vector,TFTransformer
from dataset_loader import *
from helper_functions import *
from sentiment_model import Sentiment_Model

# Inspect Function
import inspect

import time
from ml_pipeline import ML_pipeline
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Logger -- Level -- [Info]
import logging
logging.info("info")
# _______________________________________________

from sklearn.model_selection import train_test_split
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import torch
from sklearn.externals import joblib
import tensorflow_hub as hub
import tensorflow as tf
from plot_helper import *
import seaborn as sns

# _________________TFIDF_EXPERIMENTS_N-GRAMS Import_______________________
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report



##########################################################
# Collect & Melt Dataframes
##########################################################
def Collect_dframes():
    df = Dataframe_Assembly().collect_trained()
    frames={}
    path = './pickled/dataframe_fragments/{}/*'.format('pre_trained')
    for file in glob.iglob(path, recursive=True):
        for dataframe in ['imdb', 'semeval2016', 'sentiment140']:
            # frame = pd.DataFrame(joblib.load(file)[dataframe].items(), columns=['Embeddings', 'F1'])
            frame = joblib.load(file)[dataframe]
            frame = pd.DataFrame(frame)
            print('From Dataset: ', dataframe)
            print(frame.head())
            frames[dataframe] = frame
        break

    df = pd.DataFrame({'Pre-Trained Word Embedding Models':['elmo[1024]', 'ftext[300]', 'glove[300]', 'use[512]', 'w2v[300]'],
                       'IMDB':frames['imdb'].f1,
                       'Semeval2016': frames['semeval2016'].f1,
                       'Sentiment140': frames['sentiment140'].f1})
    df
    df = df.melt('Pre-Trained Word Embedding Models', var_name='Datasets', value_name='Mean F1 Score of the models SVM, Adaboost(RF), LR')


##########################################################
# Testing for validity Assembled Dataframes
##########################################################
def try_drames():
    df = pd.DataFrame.from_dict(
    {
    'Embedding Models' : ['doc2vec[50]_DBOW', 'doc2vec[50]_DM', 'glove[50]', 'word2vec[50]_CBOW', 'word2vec[50]_SG'],
    'Mean F1 Score of the models: SVM, Adaboost(RF), LR' : [0.60, 0.68, 0.69, 0.72, 0.72]
    }
    )
    sns_plot = sns.barplot(x="Mean F1 Score of the models: SVM, Adaboost(RF), LR", y="Embedding Models",
    data=df, orient='h',
    palette=sns.color_palette("muted"))
    sns_plot.figure.savefig("./images/factor_preT.png")

    dframes = []
    for dataset in ['imdb', 'semeval2016', 'sentiment140']:
        frame = Dataframe_Assembly().collect_pretrained()[dataset]
        dframe = pd.DataFrame(frame)
        dframe['dataset'] = dataset
        dframes.append(dframe)

    x = Plotter(dir='./pickled/models/svm/cross_tfidf/*').prepare_data(log=True)
    dframe_.prepare_corr_dataframe('cross_tfidf')

    dframe_ = Dataframe_Assembly()
    dframe_.collect_tfidf()


##########################################################
# TFIDF - Cross-domain Experiments Plot
##########################################################
def cross_domain():
    dframe_ = Dataframe_Assembly()
    # dframe_.prepare_corr_dataframe('tf_idf')
    frame_ = dframe_.collect_trained('tf_idf')
    frame_
    Plotter(dir=None).plot_corr_tfidf(figure='Cross-domain-mean', dataframe=frame_)
cross_domain()


##########################################################
# Word - Cloud Plot for each of the Datasets
##########################################################
sarcasm().plot_wordcloud(sarcasm=True)
sentiment140().plot_wordcloud(sarcasm=False)
semeval2016().plot_wordcloud()

##########################################################
# Pickle & Save pre-trained AllenNLP - ELMO Embeddings
##########################################################
Pipe = ML_pipeline(mode='contextual').transform()

def pipeline_contextual():
    sent_model = Sentiment_Model()

    for embedding_file in glob.iglob('./pickled/contextual_pretrained/*.pkl', recursive=True):

        dataset =  embedding_file.split('/')[-1].split('_')[-1].strip('.pkl')
        emb_name = embedding_file.split('/')[-1].split('_')[0].strip('.pkl')

        print('Loading Embeddings {} ....... Dataset  {}'.format(emb_name, dataset))

        cont_embs = joblib.load(embedding_file)
        loader = dataset_loader(dataset)

        x_emb_train, x_emb_test, y_labels_train, y_labels_test = train_test_split(
                    cont_embs, loader.load_devset()['y_labels'],
                    test_size=0.25, random_state=40, shuffle=True)

        info = dict()
        info['embedding'] = emb_name
        info['dataset'] = dataset
        transformer = None

        sent_model.classify_batch(x_emb_train, y_labels_train,
        x_emb_test, y_labels_test, info, transformer, ['svm', 'lr', 'rf'])

pipeline_contextual()


##########################################################
# Pickle & Save pre-trained AllenNLP - ELMO Embeddings
##########################################################
def pickle_elmo():
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    for dataset_file in glob.iglob('./pickled/datasets/*', recursive=True):
        Elmo = ElmoEmbedder(options_file, weight_file)

        dataset = dataset_file.split('/')[-1]
        if dataset == 'semeval2016' or dataset == 'sarcasm':
            continue

        print('Dataset: ', dataset)
        loader_ = dataset_loader(dataset)
        devset = loader_.load_devset()

        processed_sentences = list(map(lambda sentence: sentence.split(), devset['x_data']))

        print('Begin Augmenting_______')
        final_sentence_embeddings = list()
        for sentence_id, sentence in enumerate(processed_sentences):
            if sentence_id % 200 == 0 and sentence_id is not 0:
                logging.log(logging.WARNING, "Processing Sentence[No]: {}".format(sentence_id))

            sentence_embeddings = Elmo.embed_sentence(sentence)
            final_sentence_embeddings.append(np.ndarray.mean(np.ndarray.mean(sentence_embeddings, axis=0), axis=0))

        print('Saving {0} ..... Dataset {1}'.format('Allen-Elmo', dataset))
        with open('./pickled/contextual_pretrained/{}_{}.pkl'.format('elmo', dataset), 'wb') as f:
            pickle.dump(final_sentence_embeddings, f, pickle.HIGHEST_PROTOCOL)


for model_file in glob.iglob('./pickled/models/svm/*.pkl', recursive=True):
    model_name = model_file.split('/')[-1]
    print('Loading Model: ', model_name)
    model = joblib.load(model_file)[1]
    print(model)




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield list(map(lambda sentence: sentence, l[i:i + n]))

##########################################################
# Pickle & Save Contextualized Embeddings(Elmo - U.S.E.)
##########################################################
def pickle_contextual(sample=None, chunk_size=1000):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.device('/device:CPU:0')
    emb = Embedding_Vector()
    for dataset_file in glob.iglob('./pickled/datasets/*', recursive=True):
        dataset = dataset_file.split('/')[-1]
        if dataset == 'sarcasm' or dataset == 'sentiment140' or dataset == 'imdb':
            continue

        if dataset == 'imdb':
            chunk_size = 10
        else:
            chunk_size = 1000

        print('Dataset: ', dataset)
        loader_ = dataset_loader(dataset)
        devset = loader_.load_devset()
        if sample:
            class_samples = devset['x_data'][:sample] + devset['x_data'][-sample:]
            class_labels = devset['y_labels'][:sample] + devset['y_labels'][-sample:]
        else:
            print('Sample was not taken, whole dset used')
            class_samples = devset['x_data']
            class_labels = devset['y_labels']
            pass

        for cont_embeddings in ['use','elmo']:
            if cont_embeddings == 'use':
                continue

            embed = hub.Module("https://tfhub.dev/google/elmo/2") if cont_embeddings == 'elmo' else hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

            tf.logging.set_verbosity(tf.logging.ERROR)
            final_sentence_embeddings = []

            for chunk_id, chunk in enumerate(chunks(class_samples, chunk_size)):
                if chunk_id % 5 == 0 and chunk_id is not 0:
                    logging.log(logging.WARNING, "Processing chunk[no]: {}, [{}]".format(chunk_id, cont_embeddings))

                if cont_embeddings == 'elmo':
                    embeddings = embed(chunk,signature="default",as_dict=True)["elmo"]
                else:
                    embeddings = embed(chunk)

                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    raw_sentence_embeddings = sess.run(embeddings)

                for sentence_embedding in raw_sentence_embeddings:
                    final_sentence_embeddings.append(np.ndarray.mean(sentence_embedding, axis=0))


        assert(len(class_labels) == len(final_sentence_embeddings))
        devset['x_data'], devset['y_labels'] = sanity_check(final_sentence_embeddings, class_labels)

        print('Saving {} .....'.format(cont_embeddings))
        with open('./pickled/contextual_pretrained/{}_{}.pkl'.format(cont_embeddings,dataset), 'wb') as f:
            pickle.dump(devset, f, pickle.HIGHEST_PROTOCOL)


##########################################################
# T-SNE 2D & 3D reduction for K-Nearest Sentiment Neighbor
##########################################################
def fit_tsne3d():
    emb_vector = Embedding_Vector()
    emb_vector.load_trained('word2vec', 50, 'CBOW')
    model_ak = emb_vector.embeddings

    words_ak = []
    embeddings_ak = []
    #Gather All the Vocabulary Embeddings
    for word in list(model_ak.wv.vocab):
        embeddings_ak.append(model_ak.wv[word])
        words_ak.append(word)

    #Transform to TSNE[3D]
    tsne_cbow_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=32)
    embeddings_cbow_3d = tsne_cbow_3d.fit_transform(embeddings_ak)

    with open("./pickled/tsne_3d_cbow.pkl", 'wb') as file:
        pickle.dump(embeddings_cbow_3d, file, pickle.HIGHEST_PROTOCOL)
fit_tsne3d()

def tsne_plot_3d(title, label, embeddings, a=1):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.show()
import matplotlib.cm as cm

def tsne_plot_2d(label, embeddings, words=[], a=1):
    plt.figure(figsize=(6, 4))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("hhh.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()



emb = Embedding_Vector()
emb.load_trained('word2vec', '50', 'CBOW' )
embeddings = emb.embeddings
display_closestwords_tsne(embeddings, 'awful', 50)


with open("./pickled/tsne_3d_cbow.pkl", 'rb') as file:
    embeddings_cbow_3d = pickle.load(file)


tsne_plot_2d('Word2Vec TSNE[2D]', embeddings_cbow_2d, a=0.1)

with open('./pickled/dataframe_fragments/self_trained/corr_dataframe_876.pkl', 'rb') as file:
    df = pd.DataFrame(pickle.load(file).items(),  columns=['Embeddings', 'F1 score'])
    print(df.head)



os.listdir(r'./pickled/dataframe_fragments/{}/*'.format('svm'))

Plotter(dir='./pickled/models/{}/self_trained/*.pkl'.format('svm')).\
prepare_data(mode='corr_graph').plot_bar_graph(figure='{}_{}'.format('self_trained','svm'))

x='./pickled/models/svm/self_trained\glove50_sent140.pkl'
x.rsplit('/')[3]


##################################################
# Create self-trained Word Vectors
##################################################
def train_models():
    snt140 = sentiment140()
    corpus = snt140.load_dataset('sentiment140_cleaned').clean_text.tolist()

    embeddings = Embedding_Vector()
    embeddings.train('doc2vec', corpus, 50, 50, {'dm':1})
    embeddings.train('doc2vec', corpus, 50, 50, {'dm':0})

    embeddings.train('word2vec', corpus, 50, 50, {'sg': 1})
    embeddings.train('word2vec', corpus, 50, 50, {'sg': 0})
    embeddings.train('glove', corpus, dims=50, epochs=50)
    start = time.time()
    trained_sentiment140()
    end = time.time()
    print('Time Trained:', end - start)


pipeline = ML_pipeline(mode='imdb')
pipeline.transform()

pipeline = ML_pipeline(mode='sarcasm')
pipeline.transform()

pipeline = ML_pipeline(mode='semeval2017')
pipeline.transform()

pipeline = ML_pipeline(mode='sentiment140')
pipeline.transform()

Sentiment_Model().svm(x_train, y_labels_train, x_test, y_labels_test, info, trans)

Plotter(dir='./pickled/models/{}/*.pkl'.format('rf')).prepare_data(filters=['tfidf'], mode='corr_graph').plot_corr_graph(figure='{}_{}'.format('tfidf', 'rf'.rsplit('\\')[-1]))
##########################################################
# Eli5 Model Explanation on Live Missclassified Example
##########################################################
def eli5_missclassified():
    devset = semeval2017().load_devset()
    devset.keys()

    x_train, x_test, y_labels_train, y_labels_test = train_test_split(
                devset['x_data'], devset['y_labels'],
                test_size=0.25, random_state=0,
                shuffle=True)

    tf_transformer = TFTransformer()
    x_vec_train = tf_transformer.fit_transform(x_train)
    x_vec_test = tf_transformer.transform(x_test)
    import eli5
    from sklearn.svm import LinearSVC
    model = LinearSVC(C=1.0)
    model.fit(x_vec_train, y_labels_train)
    eli5.show_weights(model, vec=tf_transformer, top=30, horizontal_layout=True)
    eli5.show_prediction(model, x_test[0], vec=tf_transformer, horizontal_layout=False)
    model.predict(tf_transformer.transform(['I like cats more than i like dogs']))
    eli5.explain_prediction(model,
    'I do not like going out when it is rainy',
    vec = tf_transformer)

def eli5_exp():
    from sklearn.svm import SVC,LinearSVC
    dataset = semeval2016().load_devset()

    x_train, x_test, y_labels_train, y_labels_test = train_test_split(
                dataset['x_data'], dataset['y_labels'],
                test_size=0.25, random_state=0,
                shuffle=True)

    #best transformer parameters
    trans = TFTransformer(min_df=3, ngram_range=(1,3))

    x_train_tfvec = trans.fit_transform(x_train)
    x_test_tfvec = trans.transform(x_test)

    #best SVC parameters
    linear_svc = LinearSVC(C=0.01, penalty='l2')
    linear_svc.fit(x_train_tfvec, y_labels_train)

    eli5.explain_weights(linear_svc, vec=trans, top=(7,7))
    eli5.show_prediction(linear_svc, 'sad to not be go to disneyland tomorrow', vec=trans, horizontal_layout=False)


devsetA = semeval2016().load_devset()


##########################################################
# Sparse Experiments(TFIDF) - MNB
##########################################################

def run_tfidf(model='svm'):
    for dataset_file in tqdm(glob.iglob('./pickled/datasets/*', recursive=True), total=3, desc='Dataset Processing: '):
    dataset_name = dataset_file.split('/')[-1]

    print('__________________________________')
    print('Loading Dataset: {} ......'.format(dataset_name))
    if '.pkl' not in dataset_name:
        try:
            loader = dataset_loader(dataset_name)
        except KeyError:
            print("{} does not exist as dataset".format(dataset_name))
            sys.exit(-1)    
    
    tmp_dev = loader.load_devset()
    x_train, x_test, y_labels_train, y_labels_test = train_test_split(
                tmp_dev['x_data'], tmp_dev['y_labels'],
                test_size=0.25, random_state=0,
                shuffle=True)

    param_grid = {
                  'tftransformer__max_features' : [10, 20, 30],
                  'tftransformer__min_df': [2, 5],
                  'tftransformer__ngram_range': [(1,3)],
                  'tftransformer__norm': ['l1', 'l2', None]
                }

    estimator_ = make_pipeline(transformer, MultinomialNB())
    grid_nb=GridSearchCV(estimator=estimator_,param_grid=param_grid,cv=5, n_jobs=-1)
    grid_nb.fit(x_train,y_train)
    print('Grid Best Params: {}'.format(grid_nb.best_params_)
    y_predicted_ = nb_cv.predict(x_test)
    print(classification_report(y_labels_test, y_predicted_))
    joblib.dump("./pickled/trained_mnb/{}.pkl".format(model), grid_nb)

