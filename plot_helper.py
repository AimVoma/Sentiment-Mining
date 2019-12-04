#####################################################################
# Main Utility class that creates 2D plots utilizing seaborn and
# pyplot libs
# Graph Types: Scatterplot, Barplot, D-Frames
#####################################################################

import pandas as pd
import pickle
# import seaborn as sb
import glob
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from tqdm import tqdm
from random import randint
import os, shutil
import sys
import collections
from sklearn.externals import joblib
import seaborn as sns


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class Dataframe_Assembly:
    def __init__(self):
        pass
    def collect_trained(self, folder='self_trained'):
        dframes = list()
        mean_dframe = pd.DataFrame()
        path = './pickled/dataframe_fragments/{}/*'.format(folder)
        for file in glob.iglob(path, recursive=True):
            # dframe_file = file.rsplit('\\')[1].strip('.pkl')
            # with open(os.path.join(path,'{}.pkl'.format(dframe_file)), 'rb') as file:
            with open(file, 'rb') as file:
                frame = pd.DataFrame(joblib.load(file).items(), columns=['Embeddings', 'F1'])
                dframes.append(frame)
        try:
            assert(len(dframes) == 3)

            dframes[0].F1 = (dframes[0].F1 + dframes[1].F1 + dframes[2].F1) / 3.0
            mean_frame = dframes[0]
            mean_frame.F1 = list(map(lambda f1: round(f1, 2), mean_frame.F1))
        except IndexError as error:
            print('error accured: ', error)
        except AssertionError as asserr:
            print('Exception Error occured: ', asserr)
        return mean_frame

    #Return the Mean F1 values for each of the source -- targets domain
    def collect_tfidf(self, path = './pickled/dataframe_fragments/{}/*'.format('cross_tfidf')):
        l, fragments_ = list(), list()
        for count, file in enumerate(glob.iglob(path, recursive=True)):
            l.append(joblib.load(file))

        for key in ['imdb', 'sarcasm', 'semeval2016']:
             imdb = round((l[0][key]['imdb'] + l[1][key]['imdb'] + l[2][key]['imdb']) / 3.0, 2)
             sarcasm = round((l[0][key]['sarcasm'] + l[1][key]['sarcasm'] + l[2][key]['sarcasm']) / 3.0, 2)
             semeval2016 = round((l[0][key]['semeval2016'] + l[1][key]['semeval2016'] + l[2][key]['semeval2016']) / 3.0, 2)

             fragments_.append({key:[imdb, sarcasm, semeval2016]})

        return fragments_

    def collect_pretrained(self, dataframe=None):
        dframes = list()
        mean_dframe = pd.DataFrame()
        path = './pickled/dataframe_fragments/pre_trained/*'

        for file in glob.iglob(path, recursive=True):
            dframe_file = file.rsplit('/')[-1].strip('.pkl')
            # with open(os.path.join(path,'{}.pkl'.format(dframe_file)), 'rb') as file:
            dframes.append(joblib.load(file))

        # imdb_dframe, semeval_dframe, sentiment_dframe = list(
        # pd.DataFrame(columns=['embeddings', 'f1'])
        # for i in range(0, 3))

        imdb_dframe, semeval_dframe, sentiment_dframe = list(
        pd.DataFrame(columns=['embeddings', 'f1'])
        for i in range(0, 3))

        DFrames = {'imdb' : imdb_dframe,
        'semeval2016' : semeval_dframe,
        'sentiment140' : sentiment_dframe}

        DFrames['imdb'].f1 = round((dframes[0]['imdb']['f1'] + dframes[1]['imdb']['f1'] + dframes[2]['imdb']['f1'])/ 3.0, 2)
        DFrames['semeval2016'].f1 = round((dframes[0]['semeval2016']['f1'] + dframes[1]['semeval2016']['f1'] + dframes[2]['semeval2016']['f1'])/ 3.0, 2)
        DFrames['sentiment140'].f1 = round((dframes[0]['sentiment140']['f1'] + dframes[1]['sentiment140']['f1'] + dframes[2]['sentiment140']['f1']) / 3.0, 2)

        DFrames['imdb'].embeddings = dframes[0]['imdb']['embeddings']
        DFrames['semeval2016'].embeddings = dframes[0]['semeval2016']['embeddings']
        DFrames['sentiment140'].embeddings = dframes[0]['sentiment140']['embeddings']
        return DFrames

        # Collect all the models scores and measurements
        imdb_dframe.f1 = (dframes[0]['imdb']['f1'] + dframes[1]['imdb']['f1'] + dframes[2]['imdb']['f1']) / 3.0
        semeval_dframe.f1 = (dframes[0]['semeval2016']['f1'] + dframes[1]['semeval2016']['f1'] + dframes[2]['semeval2016']['f1']) / 3.0
        sentiment_dframe.f1 = (dframes[0]['sentiment140']['f1'] + dframes[1]['sentiment140']['f1'] + dframes[2]['sentiment140']['f1']) / 3.0

        imdb_dframe.embeddings = dframes[0]['imdb']['embeddings']
        semeval_dframe.embeddings = dframes[0]['semeval2016']['embeddings']
        sentiment_dframe.embeddings = dframes[0]['sentiment140']['embeddings']

        Plotter(dir=None).plot_bar_graph(dataframe=DFrames[dataframe])

    def prepare_corr_dataframe(self, folder):
        self.clean_old(folder)
        for model in ['svm', 'lr', 'nb']:
            Plotter(dir='./pickled/models/{0}/{1}/*.pkl'.format(model, folder)).\
            prepare_data(log=True)

    def clean_old(self, folder):
        path = r'./pickled/dataframe_fragments/{}'.format(folder)
        for dataframe_old in os.listdir(path):
            file_path = os.path.join(path, dataframe_old)
            if os.path.isfile(file_path):
                os.unlink(file_path)

class Plotter:
    def __init__(self, dir=None):
        if dir is None:
            pass
        else:
            self.dir = dir
            self.classifier = dir.split('/')[3]
            self.scores = {}
            self.corr_dataframe = None
            self.classifier_dict = {'svm': 'Linear SVC',
                                    'lr': 'Logistic Regression',
                                    'ridge': 'Ridge Classifier',
                                    'rf': 'Adaboost(Random Forest)',
                                    'nb' : 'MN Naive Bayes'}
            try:
                self.classifier = self.classifier_dict[self.classifier]
            except KeyError:
                print('Wrong Classifier')
                pass


    # emb_folder = pre_trained folder in MODELS
    def prepare_data(self, log=False):
        df_holder = None
        model = None

        assert(self.dir is not None)
        df = pd.DataFrame()
        devsets, embs, scores= [],[],[]

        pred_dict ={
                    'imdb': dict(),
                    'sarcasm': dict(),
                    'semeval2016': dict(),
                    'sentiment140': dict()
                    }

        for filename in sorted(tqdm(glob.iglob(self.dir, recursive=True), total=5, desc='Processing {} prediction'.format(self.classifier))):

            with open(filename, 'rb') as f:
                prediction = joblib.load(f)[2]

            model = filename.split('/')[-3]

            classA = prediction.split('\n')[2].split()[0]
            classB = prediction.split('\n')[3].split()[0]
            classA_F = prediction.split('\n')[2].split()[3]
            classB_F = prediction.split('\n')[3].split()[3]
            mean_f = float((float(classA_F) + float(classB_F)) / 2.0)
            score_d = dict(zip([classA, classB, 'mean'],[classA_F, classB_F, mean_f]))

            # TODO:: CHECK AGAIN THE PRED_DICT
            #Pattern Match Regex
            if filename.split('/')[-2] == 'self_trained':
                emb = file.split('_')[0] + '_' + file.split('_')[1]
                devset = file.rsplit('_')[-1]
            elif filename.split('/')[-2] == 'cross_tfidf':
                file = filename.split('/')[-1]

                source_domain = file.rsplit('[')[1].rsplit(']')[0]
                target_domain = file.rsplit('_')[1].strip('.pkl')

                try:
                    pred_dict[source_domain][target_domain] = score_d['mean']
                    pred_dict['model'] = model
                except KeyError:
                    raise Warning('TF-IDF Domain not found, {0}-{1} not found'.format(source_domain, target_domain))

                except AttributeError:
                    # Filter not found, continue
                    print('No Match File:', file)
                    continue
            else:
                df_holder = {
                'imdb': pd.DataFrame(columns=['embeddings', 'f1']),
                'sentiment140': pd.DataFrame(columns=['embeddings', 'f1']),
                'semeval2016': pd.DataFrame(columns=['embeddings', 'f1'])
                }
                dframe_size = len(df_holder[devset])
                # Add as last element, Increase index & rearrange
                df_holder[devset].loc[dframe_size] = [emb, score_d['mean']]

        if log:
            path = r'./pickled/dataframe_fragments/'
            file = os.path.join(path,'{0}/{1}.pkl'.format('cross_tfidf', 'dframe_{}'.format(model)))
            joblib.dump(pred_dict, file)

        if mode is 'corr_graph':
            sorted_pred_dict = {}
            # self_trained = collections.OrderedDict(sorted(self_trained.items()))
            for key,values in pred_dict.items():
                sorted_pred_dict[key] = collections.OrderedDict(sorted(values.items()))

            for keys, values in sorted_pred_dict.items():
                print(keys, values)
            self.corr_dataframe = pd.DataFrame(sorted_pred_dict)

            if log:
                path = r'./pickled/dataframe_fragments/'
                with open(os.path.join(path,'{0}/{1}.pkl'.format(emb_folder, 'corr_dataframe_' + str(randint(100, 999)))), 'wb') as f:
                    pickle.dump(self.corr_dataframe, f, pickle.HIGHEST_PROTOCOL)

        return pred_dict

    def plot_factor_preT(self):
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

        df = df.melt('Pre-Trained Word Embedding Models', var_name='Datasets', value_name='Mean F1 Score of the models SVM, Adaboost(RF), LR')
        #alternative for pandas < 0.20.0
        #df = pd.melt(df, 'X_Axis', var_name='cols',  value_name='vals')
        df["Mean F1 Score of the models SVM, Adaboost(RF), LR"] = list(map(lambda x : round(x, 2),df["Mean F1 Score of the models SVM, Adaboost(RF), LR"]))
        print(df.head)
        sns_plot = sns.factorplot(x="Pre-Trained Word Embedding Models", y="Mean F1 Score of the models SVM, Adaboost(RF), LR", hue='Datasets', data=df, palette=sns.color_palette("husl", 10))
        sns_plot.savefig("./images/factor_preT.png")
        return df

    def plot_bar_graph(self, figure='figure', dataframe=None):
        plt.close('all')
        plt.rcParams['figure.figsize'] = (5,3)
        # self.corr_dataframe = self.corr_dataframe.sort_values(['dataset'],ascending=True)
        # self.corr_dataframe = self.corr_dataframe.set_index('dataset')

        fig = plt.figure() # Create matplotlib figure
        ax = fig.add_subplot(111) # Create matplotlib axes

        # width = .15
        dataframe.reset_index()
        dataframe.dropna(how='all', axis=1, inplace=True)
        dataframe.set_index('embeddings', inplace=True)
        my_colors = 'gkymc'
        dataframe.plot(ax=ax, kind='barh', xticks=np.arange(0.0, 1.0, 0.1), color=my_colors, legend=False, title='Training Corpus: Sentiment140')
        ax.grid(None, axis=0)
        ax.set_ylabel('Self-Trained Word Embeddings')
        ax.set_xlabel('Mean F1 Score of the models: SVM, Adaboost(RF), LR')
        #
        # leg1=ax.legend(bbox_to_anchor=(1.05, 1.05),
        # title='TFIDF Performance', title_fontsize=10.5)

        plt.savefig('./plots/{}'.format(figure), dpi=fig.dpi)

    def plot_corr_tfidf(self, figure='figure', dataframe=None):

        if dataframe is None:
            assert(self.corr_dataframe is not None)
            graph_type = 'single'
            dataframe = self.corr_dataframe
        else:
            graph_type = 'mean'

        print('Plotting Correlation Graph of {}'.format(graph_type))
        print(dataframe.head())

        plt.close('all')
        plt.rcParams['figure.figsize'] = (5,3)
        # self.corr_dataframe = self.corr_dataframe.sort_values(['dataset'],ascending=True)
        # self.corr_dataframe = self.corr_dataframe.set_index('dataset')

        fig = plt.figure() # Create matplotlib figure
        ax = fig.add_subplot(111) # Create matplotlib axes
        # width = .15
        dataframe.reset_index()
        dataframe.dropna(how='all', axis=1, inplace=True)
        dataframe.plot(ax=ax, kind='barh', xticks=np.arange(0.0, 1.0, 0.1), colormap='winter')
        plt.savefig('./plots/{}'.format(figure), dpi=fig.dpi)

        # self.corr_dataframe.plot(ax=ax, kind='barh', figsize=(600,400))

        # self.corr_dataframe.self_trained.plot(kind='bar',color='DarkGreen',ax=ax,width=width, position=0)
        # self.corr_dataframe.pre_trained.plot(kind='bar',color='DarkBlue', ax=ax,width=width,position=1)

        ax.grid(None, axis=0)
        ax.set_ylabel('TF-IDF Vectorizer Fitted On Datasets')

        if graph_type == 'single':
            ax.set_xlabel('F1 Score, {}'.format(self.classifier))
        else:
            ax.set_xlabel('Mean F1 Score of the models: SVM, MNB, LR')

        leg1=ax.legend(bbox_to_anchor=(1.05, 1.05),
        title='TFIDF Performance', title_fontsize=10.5)

        ax.add_artist(leg1)
