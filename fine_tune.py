###########################################################
# Fine-Tuning pre-trained Elmo model weights with keras GPU
# on Sentiment dataset
###########################################################

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np
import pickle
from tensorflow.python.client import device_lib
from sklearn.metrics import classification_report

device_lib.list_local_devices()

# Configure GPU process Memory for easy load
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
backend.set_session(session)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True)

  train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      "aclImdb", "test"))

  return train_df, test_df

train_df, test_df = download_and_load_datasets()
train_df.head()


# ________________________________________________ SAVE __________________________________
with open('./pickled/'+ 'aclImdb_train' + '.pkl', 'wb') as f:
    pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)

with open('./pickled/'+ 'aclImdb_test' + '.pkl', 'wb') as f:
    pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)


# ____________________________________________________ LOAD _______________________________________

with open('./pickled/'+ 'aclImdb_train' + '.pkl', 'rb') as f:
    train_df = pickle.load(f)

with open('./pickled/'+ 'aclImdb_test' + '.pkl', 'rb') as f:
    test_df = pickle.load(f)

train_df.head()
test_df.head()

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=False
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += backend.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(backend.squeeze(backend.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return backend.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

# Function to build model
def build_model():
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  dense = layers.Dense(256, activation='relu')(embedding)
  pred = layers.Dense(1, activation='sigmoid')(dense)

  model = Model(inputs=[input_text], outputs=pred)

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()

  return model

# Create datasets (Only take up to 150 words for memory)
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:50]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['polarity'].tolist()

test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:50]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['polarity'].tolist()


# Build and fit
model = build_model()
model.fit(train_text,
          train_label,
          verbose=1,
          validation_data=(test_text, test_label),
          epochs=2,
          batch_size=5)

model.save('ElmoModel.h5_2Epochs_non-trainable')

pre_save_preds = model.predict(test_text[0:10]) # predictions before we clear and reload model
pre_save_preds
# Clear and load model
model = None
model = build_model()
model.load_weights('ElmoModel.h5_2Epochs_trainable')

post_save_preds = model.predict(test_text[0:100]) # predictions after we clear and reload model
all(pre_save_preds == post_save_preds) # Are they the same?

# ____________________________________________ TESTING __________________________________________
text_n = test_df[test_df['polarity'] == 0][:1000]
text_p = test_df[test_df['polarity'] == 1][:1000]

n_text = text_n['sentence'].tolist()
n_sent = text_n['polarity'].tolist()

p_text = text_p['sentence'].tolist()
p_sent = text_p['polarity'].tolist()

sentences,sentiment = [],[]

sentences = n_text + p_text
sentiment = n_sent + p_sent

sentences = [' '.join(t.split()[0:125]) for t in sentences]
sentences = np.array(sentences, dtype=object)[:, np.newaxis]


pre_save_preds = []
chunks = [sentences[x:x+5] for x in range(0, len(sentences), 5)]

model.predict(sentences[3])

for chunk in chunks:
    pre_save_preds.append(model.predict(chunk))

nn_pred=[]
for chunk in pre_save_preds:
    for pred in chunk:
        nn_pred.append(1 if pred>=0.5 else 0)

print(classification_report(y_true=sentiment, y_pred=nn_pred))
