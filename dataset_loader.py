##########################################################
# Main Utilization Module for Loading/Saving On Demand
# [Train/Test] Datasets with preprocessing & cleaning
# Datasets: Sentiment140, ImDB, Sarcasm, Kaggle
##########################################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ______________________WARNINGS_______________________________
import os
import numpy as np
import pandas as pd
import random
import re, pickle
import zipfile
from matplotlib import pyplot
from abc import abstractmethod
from tqdm import tqdm
import sys

from helper_functions import *
stopWords = None
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

# Dataset Hash Dictionary
def dataset_loader(dataset):
    try:
        return {
              'imdb' : imdb(),
              'sarcasm' : sarcasm(),
              'semeval2016' : semeval2016()
              }[dataset]
    except KeyError:
        print('Dataset: {}, Not Found!'.format(dataset))

# Main Loader
class loader_:
    @classmethod
    def assign_dataset(cls, dataset):
        cls.dataset = dataset

    @classmethod
    def assign_devset(cls, devset):
        cls.devset = devset

    @classmethod
    def load_devset(cls):
        with open('./pickled/datasets/{}/{}.pkl'.format(cls.__name__,'dev_set'), 'rb') as f:
            return pickle.load(f)

    @classmethod
    def save_devset(cls):
        with open('./pickled/datasets/{}/{}.pkl'.format(cls.__name__,'dev_set'), 'wb') as f:
            pickle.dump(cls.devset, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_dataset(cls, name):
        with open('./pickled/datasets/{}/{}.pkl'.format(cls.__name__, name), 'rb') as f:
            return pickle.load(f)

    @classmethod
    def save_dataset(cls, name):
        with open('./pickled/datasets/{}/{}.pkl'.format(cls.__name__, name), 'wb') as f:
            pickle.dump(cls.dataset, f, pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def create_devset(cls, size, *args):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def polarity_distribution(self):
        raise NotImplementedError("Please Implement this method")

    def plot_wordcloud(self, sarcasm=False):
        devset = pd.DataFrame(self.load_devset())

        #Define Stop Words
        stopwords = set(STOPWORDS)
        stopwords.remove('no')
        stopwords.remove('not')

        stopwords.update(['tomorrow','will', 'th',
        'st', 'see','go', 'may','sunday', 'today',
        'saturday', 'tonight',
        'today', 'think', 'say', 'day', 'now',
        'make', 'want', 'time', 'know', 'film', 'movie'])
        def polarity_plot(polarity_, dataset, sarcasm=False):
            sentences = devset['x_data'][devset.y_labels == polarity_].tolist()
            text = ' '.join(map(str, sentences))

            mask = np.array(Image.open("images/positive_mask.png")) if polarity_ else np.array(Image.open("images/negative_mask.png"))
            if not sarcasm:
                polarity = 'positive' if polarity_ else 'negative'
            else:
                polarity = 'sarcastic' if polarity_ else 'neutral'
                mask = np.array(Image.open("images/negative_mask.png")) if polarity_ else np.array(Image.open("images/positive_mask.png"))


            wordcloud_model = WordCloud(stopwords=stopwords, background_color= 'white', mode="RGBA", max_words=1000, mask=mask).generate(text)
            image_colors = ImageColorGenerator(mask)
            # plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
            plt.imsave('images/{0}_{1}.png'.format(dataset, polarity), wordcloud_model.recolor(color_func=image_colors))
        if self.__class__.__name__ == 'sentiment140':
            polarity_plot(polarity_ = 4, dataset = self.__class__.__name__, sarcasm=sarcasm)
        else:
            polarity_plot(polarity_ = 1, dataset = self.__class__.__name__, sarcasm=sarcasm)

        polarity_plot(polarity_ = 0, dataset = self.__class__.__name__, sarcasm=sarcasm)

    def plot_frequency(self, pos_freq, neg_freq, labels, bins, dataset, position):
        assert(labels is not None and dataset is not None and bins is not None)
        bins = bins
        plt.style.use('seaborn-deep')
        plt.hist([pos_freq, neg_freq], bins, label=labels, histtype='bar', color=['mediumseagreen','indianred'])
        plt.legend(loc='upper right') if position=='right' else plt.legend(loc='upper left')
        plt.xlabel('Word Density')
        plt.ylabel('Sentence Count')
        plt.title('{}'.format(dataset))
        plt.savefig('images/FPlot_{}'.format(self.__class__.__name__))
        plt.show()

class kaggle(loader_):
    def __init__(self):
        self.dset_loc = './Datasets/twitter_kaggle'
        self.clean_reviews = list()
        self.reviews = list()

        self.df = pd.read_csv('{}/{}'.format(self.dset_loc, 'train.csv'),
            encoding='latin-1',
            names=['ItemID', 'Sentiment', 'SentimentText'],
            usecols=['Sentiment', 'SentimentText']
        )

    def create_devset(self, size, *args):
        pos_df = self.df[self.df.Sentiment == '1'][:size]
        neg_df = self.df[self.df.Sentiment == '0'][:size]

        pos_df['clean_tweet'] = pos_df.SentimentText.apply(clean_text)
        neg_df['clean_tweet'] = neg_df.SentimentText.apply(clean_text)

        pos_df['word_count'] = pos_df.clean_tweet.apply(word_count)
        neg_df['word_count'] = neg_df.clean_tweet.apply(word_count)

        pos_df = pos_df.drop(pos_df[pos_df.word_count < 3].index)


        pos_df = pos_df[pos_df.word_count > 3]
        neg_df = neg_df[neg_df.word_count > 3]

        if pos_df.shape[0] > size:
            print('Pos Is bigger')
            pos_df = pos_df[:size]

        if neg_df.shape[0] > size:
            print('Neg Is bigger')
            neg_df = neg_df[:size]

        Y_labels = pos_df.Sentiment.tolist() + neg_df.Sentiment.tolist()
        X_data = pos_df.clean_tweet.tolist() + neg_df.clean_tweet.tolist()
        word_count = pos_df.word_count.tolist() + neg_df.word_count.tolist()

        dev_dict = {"x_data": X_data,
                "y_labels": Y_labels,
                "word_count" : word_count}

        super().assign_devset(dev_dict)
        self.save_devset()

    def polarity_distribution(self):
        devset = pd.DataFrame(self.load_devset())
        devset['word_count'] = devset.x_data.apply(word_count)

        sarc_freq = devset['word_count'][devset.Sentiment == 1]
        neut_freq = devset['word_count'][devset.Sentiment == 0]

        self.plot_frequency(sarc_freq, neut_freq,
                            labels=['sarcastic', 'neutral'],
                            dataset = 'kaggle')


class imdb(loader_):
    def __init__(self):
        self.dset_loc = './Datasets/imdb_reviews/'
        self.REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        self.clean_reviews = list()
        self.reviews = list()

    def preprocess_reviews(self):
        self.reviews = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in self.reviews]
        self.reviews = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in self.reviews]

        with open('./pickled/datasets/imdb/{}.pkl'.format('reviews2'), 'wb') as f:
            pickle.dump(self.reviews, f, pickle.HIGHEST_PROTOCOL)


    def create_devset(self, size=2500, *args):
        with open('./pickled/datasets/imdb/{}.pkl'.format('reviews1'), 'rb') as file:
            reviews = pickle.load(file)

        assert(len(reviews) == 25000)

        pos_reviews = reviews[:size]
        neg_reviews = reviews[-size:]

        pos_labels = [1 for value in pos_reviews]
        neg_labels = [0 for value in neg_reviews]

        df = pd.DataFrame(columns=['sentences', 'sentiment'])

        df.sentences = pos_reviews + neg_reviews
        df.sentiment = pos_labels + neg_labels
        df['word_count'] = df.sentences.apply(word_count)

        df['sentences'] = df.sentences.apply(clean_text)

        df = df.drop(df[df.word_count < 3].index)

        dev_dict = {"x_data": df.sentences.tolist(),
                "y_labels": df.sentiment.tolist(),
                'word_count': df.word_count.tolist()}

        super().assign_devset(dev_dict)
        self.save_devset()

    def polarity_distribution(self):
        devset = pd.DataFrame(self.load_devset())

        if 'word_count' not in devset:
            devset['word_count'] = devset.x_data.apply(word_count)

        pos_freq = devset['word_count'][devset.y_labels == 1]
        neg_freq = devset['word_count'][devset.y_labels == 0]

        self.plot_frequency(pos_freq, neg_freq,
                            ['Positive Sentences', 'Negative Sentences'],
                            range(20, 120, 7), 'IMDB Movie Reviews', 'left')

class semeval2016(loader_):
    def __init__(self):
        self.dname = 'twitter_dataframe'
        self.dloc = './Datasets/Twitter_Dataset[2013_2016].txt'

    def extract_dataset(self):
        df = pd.read_csv(self.dloc ,index_col=None, sep='\t', header=None, names=['id','sentiment','text'])
        df = df.drop_duplicates()

        df['text']= df['text'].apply(clean_text)
        df['word_count'] = df['text'].apply(word_count)
        df = df.drop('id', 1)
        df = df.drop(df[df.word_count < 3].index)

        super().assign_dataset(df)
        self.save_dataset(self.dname)
        return self

    def create_devset(self, split=7500):
        df = self.load_dataset(self.dname)

        if split == 0:
            return df.text.tolist(), df.sentiment.tolist()
        else:
            df_pos = df[df.sentiment == 'positive'][:split]
            df_neg = df[df.sentiment == 'negative'][:split]

            pos_labels = [1 for value in df_pos['sentiment'].values.tolist()]
            neg_labels = [0 for value in df_pos['sentiment'].values.tolist()]

            pos_text = df_pos['text'].values.tolist()
            neg_text = df_neg['text'].values.tolist()

            Y_labels = pos_labels + neg_labels
            X_data = pos_text + neg_text
            word_count = df_pos.word_count.tolist() + df_neg.word_count.tolist()

            dev_dict = {"x_data": X_data,
                    "y_labels": Y_labels,
                    "word_count": word_count}

            super().assign_devset(dev_dict)
            self.save_devset()

    def polarity_distribution(self):
        devset = pd.DataFrame(self.load_devset())
        devset['word_count'] = devset.x_data.apply(word_count)

        pos_freq = devset['word_count'][devset.y_labels == 1]
        neg_freq = devset['word_count'][devset.y_labels == 0]

        self.plot_frequency(pos_freq, neg_freq,
                            ['Positive Sentences', 'Negative Sentences'],
                            range(3, 35, 2), 'Twitter: SemEval[2016]', 'left')


class sarcasm(loader_):
    def __init__(self):
        self.dloc = './Datasets/{}/train-balanced-sarcasm.csv'.format(self.__class__.__name__)
        self.df = None
        self.devset = None

    def clean_dataset(self):
        try:
            self.df = self.load_devset('sarcasm_dataframe')
        except:
            self.df = pd.read_csv(self.dloc, encoding='latin1', usecols=['comment', 'label'])

        self.df.dropna(inplace=True)
        self.df['clean_comment'] = self.df['comment'].apply(clean_text)
        self.df.drop(columns=['comment'], axis=1)
        self.df['word_count'] = self.df['clean_comment'].apply(word_count)
        self.df = self.df[self.df.word_count > 3]

        super().assign_dataset(self.df)
        self.save_dataset('dataframe_sarcasm[cleaned]')

    def create_devset(self, size, *args):
        try:
            self.df = self.load_dataset('sarcasm_cleaned')
        except:
            print("Dataset dataframe_sarcasm[cleaned] do not exist!!!")
            sys.exit(-1)

        rand = random.randrange(1000, 5000) * random.randrange(1,30)

        df_neut = self.df[self.df.label == 0][rand: rand + size]
        df_sarc = self.df[self.df.label == 1][rand: rand + size]

        try:
            if args and args[0]['stop_words']:
                global stopWords
                with open('./stop_words_custom.txt', encoding="Latin1") as file:
                    s_words = file.readlines()
                    stopWords =  list(map(lambda word: re.sub('\n',  '', word), s_words))

                df_neut['clean_comment'] = df_neut['clean_comment'].apply(rstop_words)
                df_sarc['clean_comment'] = df_sarc['clean_comment'].apply(rstop_words)
        except KeyError:
            pass

        Y_labels = df_neut.label.values.tolist() + df_sarc.label.values.tolist()
        X_data = df_neut.clean_comment.values.tolist() + df_sarc.clean_comment.values.tolist()
        word_count = df_neut.word_count.tolist() + df_sarc.word_count.tolist()

        dev_dict = {"x_data": X_data,
                "y_labels": Y_labels,
                "word_count": word_count}

        super().assign_devset(dev_dict)
        self.save_devset()

    def polarity_distribution(self):
        devset = pd.DataFrame(self.load_devset())
        devset['word_count'] = devset.x_data.apply(word_count)

        sarc_freq = devset['word_count'][devset.y_labels == 1]
        neut_freq = devset['word_count'][devset.y_labels == 0]

        self.plot_frequency(sarc_freq, neut_freq,
                            ['Sarcastic Sentences', 'Neutral Sentences'],
                            range(3, 34, 2), 'Twitter: Sarcasm', 'right')

class sentiment140(loader_):
    def __init__(self, file_path="./Datasets/sentiment140.zip", file='training.1600000.processed.noemoticon.csv'):
        self.file_path = file_path
        self.file = file
        self.dataset = None
        self.dev_set = None

    def open_zipped(self):
        try:
            zf = zipfile.ZipFile(self.file_path)
            df = pd.read_csv(zf.open(self.file),
                encoding='latin-1',
                names=['target', 'id', 'date', 'flag', 'user', 'text'],
                usecols=['target', 'text']
            )
        except IOError as e:
            logger.exception("File I/O error")

        return df

    def init_dataset(self, verbose=False):
        assert(self.file_path is not None)
        positives, negatives = list(),list()
        # Open file path
        df = self.__open_zipped()

        df['clean_text'] = df['text'].apply(simple_clean)
        df = df.drop(columns = ['text'])

        df = df[df['clean_text'] != '']
        df['word_count'] = df.clean_text.apply(word_count)

        self.assign_dataset(df)
        self.save_dataset('sentiment140_cleaned')

    def create_corpus(self, test_chunk, filter=3):
        self.dataset = self.load_dataset('sentiment140_cleaned')
        assert(self.dataset is not None)

        if filter != 0:
            self.dataset = self.dataset[self.dataset.word_count > filter]

        pos_frame = self.dataset[self.dataset.target == 4]
        neg_frame = self.dataset[self.dataset.target == 0]

        dev_neg = neg_frame[-test_chunk:]
        dev_pos = pos_frame[-test_chunk:]

        pos_frame = pos_frame[:-test_chunk]
        neg_frame = neg_frame[:-test_chunk]

        self.corpus = pos_frame.clean_text.values.tolist() + neg_frame.clean_text.values.tolist()
        self.dev_set = [dev_neg, dev_pos]

        super().assign_dataset(self.corpus)
        self.save_dataset('sent140_corpus')

        return self

    def create_devset(self, _size, *args):
        if self.dev_set == None:
            self.dev_set = self.load_dataset('sentiment140_cleaned')

        X_data, Y_labels = [],[]
        neg_,pos_ = 0,4

        try:
            if args[0]['seed']:
                chunk = random.choice(range(1, 4, 1)) * 10000
        except KeyError:
            chunk=0
        except IndexError:
            chunk=0

        text_n = self.dev_set[self.dev_set.target==neg_].clean_text.tolist()
        text_p = self.dev_set[self.dev_set.target==pos_].clean_text.tolist()

        pol_n = self.dev_set[self.dev_set.target==neg_].target.tolist()
        pol_p = self.dev_set[self.dev_set.target==pos_].target.tolist()

        text = text_n[chunk:chunk + _size] + text_p[chunk: chunk + _size]
        polarity = pol_n[chunk:chunk + _size] + pol_p[chunk: chunk + _size]

        dev_dict = {"x_data": text,
                "y_labels": polarity}

        super().assign_devset(dev_dict)
        self.save_devset()

    def polarity_distribution(self):
        self.devset = pd.DataFrame(self.load_devset())
        self.devset['word_count'] = self.devset.x_data.apply(word_count)

        pos_freq = self.devset['word_count'][self.devset.y_labels == 4]
        neg_freq = self.devset['word_count'][self.devset.y_labels == 0]
        self.plot_frequency(pos_freq, neg_freq,
                            ['Positive Sentences', 'Negative Sentences'],
                            range(3, 35, 2), 'Twitter: Sentiment140', 'right')
