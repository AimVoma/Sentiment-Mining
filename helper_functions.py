######################################################################
#Main Python Module that contains Contractions and NLP operations
#but also T-SNE closest(k) reduction algorithm for SO Analysis.
#####################################################################
import warnings, re
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ______________________WARNINGS_______________________________
import codecs, unidecode
from sklearn.manifold import TSNE
import numpy as np
import eli5
import matplotlib.pyplot as plt
import gensim

import spacy
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])



contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot",
                   "can't've": "cannot have", "could've": "could have",
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not",
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did",
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                   "I'll've": "I will have","I'm": "I am", "I've": "I have",
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                   "i'll've": "i will have","i'm": "i am", "i've": "i have",
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                   "it'll": "it will", "it'll've": "it will have","it's": "it is",
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                   "this's": "this is","lol": "laughing",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is",
                   "there'd": "there would", "there'd've": "there would have","there's": "there is",
                   "here's": "here is", "lol":"laugh", "i'll":"I will",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                   "they'll've": "they will have", "they're": "they are", "they've": "they have",
                   "to've": "to have", "wasn't": "was not", "we'd": "we would",
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                   "we're": "we are", "we've": "we have", "weren't": "were not",
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                   "what's": "what is", "what've": "what have", "when's": "when is",
                   "when've": "when have", "where'd": "where did", "where's": "where is",
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                   "who's": "who is", "who've": "who have", "why's": "why is","nub": "noob",
                   "why've": "why have", "will've": "will have", "won't": "will not",
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                   "you'll've": "you will have", "you're": "you are", "you've": "you have",
                   "fck": "fuck", "idk":"I do not know","wtf": "what the fuck",
                   "cause": "because", "noob": "amateur", "afaik":"as far as I know", "atm": "at the moment",
                   "bs":"bull shit", "btw":"by the way", "ftw": "for the win", "fyi":"for your information",
                   "gfu":"good for you", "gr8":"great", "gratz":"congratulations", "idc" :"i do not care",
                   "kappa": "sarcasm", "lmao":"laughing", "nvm":"never mind",
                   "ofc":"of course", "plz":"please", "smd":"suck my dick", "thx":"thank you",
                   "rly":"really", "omg":"oh my god", "tldr":"too long to read", "ffs":"for fuck sake",
                   "b4":"before", "afaik":"as far as i know", "ty":"thank you", "wtv":"whatever"}


def clean_text(text, verbose = False):
    if isinstance(text, float):
        return ''
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)

    apostrophe_handled = re.sub("’", "'", decoded)
    apostrophe_handled_lower = ' '.join([token.lower() for token in apostrophe_handled.split()])
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled_lower.split()])

    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@') or str(t).startswith('#'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                letters_only = re.sub("(\d+\w.?|\w+\d.?)", '', sc_removed)
                if len(letters_only) > 1:
                    final_tokens.append(letters_only)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)

    return spell_corrected


def simple_clean(text):
    if not isinstance(text, str):
        return ''

    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)

    apostrophe_handled = re.sub("’", "'", decoded)
    apostrophe_handled_lower = ' '.join([token.lower() for token in apostrophe_handled.split()])
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled_lower.split()])

    return ' '.join(gensim.utils.simple_preprocess(expanded))


def rstop_words(sentence):
    if isinstance(sentence, float):
        return ''

    cleaned_sentence = []
    for word in sentence.split():
        if word not in stopWords:
            cleaned_sentence.append(word)
        else:
            pass
    return ' '.join(cleaned_sentence)

def rstop_words_all(sentences):
    cleaned_sentences,cleaned_sentence = list(), list()

    for sentence in sentences:
        cleaned_sentence = []
        for word in sentence.split():
            if word not in s_words:
                cleaned_sentence.append(word)
            else:
                pass
        cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences

def word_count(sentence):
    if not isinstance(sentence, str):
        return len('')
    return len(sentence.split())


def initialize_dataset(split, dset, whole_dset = False, file=False):
    if file:
        with open('./pickled/' + dset + '.pkl', 'rb') as f:
            df = pickle.load(f)
    else:
        df = dset

    if whole_dset:
        return df.text.tolist(), df.sentiment.tolist()

    df_pos = df[df.sentiment == 'positive'][:split]
    df_neg = df[df.sentiment == 'negative'][:split]

    pos_labels = [1 for value in df_pos['sentiment'].values.tolist()]
    neg_labels = [0 for value in df_pos['sentiment'].values.tolist()]

    pos_text = df_pos['text'].values.tolist()
    neg_text = df_neg['text'].values.tolist()

    Y_labels = pos_labels + neg_labels
    X_data = pos_text + neg_text

    return X_data, Y_labels


def display_closestwords_tsne(model, word, dims):
    arr = np.empty((0,int(dims)), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

def sanitize_embeddings(x_emb, y_label):
    to_remove = []
    print("Before: embeddings - {}, labels - {}".format(str(len(x_emb)), str(len(y_label))))
    for count,emb in enumerate(x_emb):
        if isinstance(emb, np.float64):
            to_remove.append(count)

    if to_remove:
        # BEWARE_________ np.delete
        print('Removing Unlabeled Embeddings, count: {}'.format(str(len(to_remove))))
        x_emb = np.delete(x_emb, to_remove, 0)
        for index in sorted(to_remove, reverse=True): del y_label[index]

        print("After: embeddings - {}, labels - {}".format(str(len(x_emb)), str(len(y_label))))

        assert(len(x_emb) == len(y_label))
        return list(x_emb), y_label
    else:
        return x_emb, y_label
