"""
This implementation is a Convolutional Neural Network for sentence classification.

It uses the same preprocessing of Kim et al., EMNLP 2014, 'Convolutional Neural Networks for Sentence Classification ' (https://github.com/yoonkim/CNN_sentence).

Run the code:
1) Run 'python preprocess.py'. This will preprocess.py the dataset and create the necessary pickle files in the pkl/ folder.
2) Run this code via: python cnn.py


Code was tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 & TensorFlow 1.2.1
- Keras 2.0.5

Data structure:
To run this network / to run a sentence classification using CNNs, the ShareData must be in a certain format.
The list train_sentences containts the different sentences of your training ShareData. Each word in the training ShareData is converted to
the according word index in the embeddings matrix. An example could look like:
[[1,6,2,1,5,12,42],
 [7,23,56],
 [35,76,23,64,17,97,43,62,47,65]]

Here we have three sentences, the first with 7 words, the second with 3 words and the third with 10 words.
As our network expects a matrix as input for the mini-batchs, we need to bring all sentences to the same length. This is a requirement
of Theano to run efficiently.  For this we use the function 'sequence.pad_sequences', which adds 0-padding to the matrix. The list/matrix will look after the padding like this:
[[0,0,0,1,6,2,1,5,12,42],
 [0,0,0,0,0,0,0,7,23,56],
 [35,76,23,64,17,97,43,62,47,65]]

To make sure that the network does not interpret 0 as some word, we set the embeddings matrix (word_embeddings) such that the 0-column only contains 0. You can check this by outputting word_embeddings[0].


Our labels (y_train) are a 1-dimensional vector containing the binary label for out sentiment classification example.

This code uses the functional API of Keras: https://keras.io/getting-started/functional-api-guide/

It implements roughly the network proposed by Kim et al., Convolutional Neural Networks for Sentence Classification, using convolutions
with several filter lengths.

Performance after 5 epochs:
Dev-Accuracy: 79.09% (loss: 0.5046)
Test-Accuracy: 77.44% (loss: 0.5163)
"""
from __future__ import print_function

from pathlib import Path

import numpy as np

import my_path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

np.random.seed(1337)  # for reproducibility

import gzip
import sys

if (sys.version_info > (3, 0)):
    import pickle as pkl
else:  # Python 2.7 imports
    import cPickle as pkl

import keras
import evaluation
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence

roc_list = []


def wordIdxLookup(word, word_idx_map):
    if word in word_idx_map:
        return word_idx_map[word]


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            roc_list.append(score)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch + 1, score))


data_path = Path(my_path.pkl_dir, 'data.pkl.gz')
emotion_path = Path(my_path.pkl_dir, 'emotion.pkl.gz')

data = pkl.load(gzip.open(data_path, "rb"))
emotionDict = pkl.load(gzip.open(emotion_path, "rb"))
print("ShareData loaded!")

train_labels = data['train']['labels']
train_sentences = data['train']['sentences']

dev_labels = data['dev']['labels']
dev_sentences = data['dev']['sentences']

test_labels = data['test']['labels']
test_sentences = data['test']['sentences']

word_embeddings = data['wordEmbeddings']

neg_weights = emotionDict['neg']
pos_weights = emotionDict['pos']
n_out = 2
# :: Find the longest sentence in our dataset ::
max_sentence_len = 0
for sentence in train_sentences + dev_sentences + test_sentences:
    max_sentence_len = max(len(sentence), max_sentence_len)

print("Longest sentence: %d" % max_sentence_len)

y_train = np.array(train_labels)
y_dev = np.array(dev_labels)
y_test = np.array(test_labels)

X_train = sequence.pad_sequences(train_sentences, maxlen=max_sentence_len)
X_dev = sequence.pad_sequences(dev_sentences, maxlen=max_sentence_len)
X_test = sequence.pad_sequences(test_sentences, maxlen=max_sentence_len)

print('X_train shape:', X_train.shape)
print('X_dev shape:', X_dev.shape)
print('X_test shape:', X_test.shape)

#  :: Create the network ::

print('Build model...')

# set parameters:
batch_size = 50

nb_filter = 50
filter_lengths = [1, 3, 5]
hidden_dims = 16
nb_epoch = 15

words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')

# Our word embedding layer
wordsEmbeddingLayer = Embedding(word_embeddings.shape[0],
                                word_embeddings.shape[1],
                                weights=[word_embeddings],
                                trainable=False)

words = wordsEmbeddingLayer(words_input)

# Now we add a variable number of convolutions
words_convolutions = []
for filter_length in filter_lengths:
    words_conv = Convolution1D(filters=nb_filter,
                               kernel_size=filter_length,
                               padding='same',
                               activation='tanh',
                               strides=1)(words)

    words_conv = GlobalMaxPooling1D()(words_conv)
    # words_conv = Dropout(0.25)(words_conv)

    words_convolutions.append(words_conv)

output = concatenate(words_convolutions)

# We add a vanilla hidden layer together with dropout layers:
output = Dropout(0.5)(output)

#########################################################################3
cnn_word_filter_neg_out = Convolution1D(filters=neg_weights[0].shape[2],
                                        filter_length=1,
                                        border_mode='same',
                                        # activation='tanh',
                                        # subsample_length=1,
                                        weights=neg_weights,
                                        trainable=False)(words)

cnn_word_filter_neg_out = Lambda(lambda x: (-1) * x)(cnn_word_filter_neg_out)
cnn_word_filter_neg_out = GlobalMaxPooling1D()(cnn_word_filter_neg_out)
cnn_word_filter_neg_out = Dropout(0.5)(cnn_word_filter_neg_out)
##################################################3
cnn_word_filter_pos_out = Convolution1D(filters=pos_weights[0].shape[2],
                                        filter_length=1,
                                        border_mode='same',
                                        # activation='tanh',
                                        # subsample_length=1,
                                        weights=pos_weights,
                                        trainable=False)(words)

cnn_word_filter_pos_out = GlobalMaxPooling1D()(cnn_word_filter_pos_out)
cnn_word_filter_pos_out = Dropout(0.5)(cnn_word_filter_pos_out)

#######################################
#######################################
output = concatenate([output, cnn_word_filter_neg_out, cnn_word_filter_pos_out])

#########################################################################3333
output = Dense(hidden_dims, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001))(output)
# output = GlobalMaxPooling1D()(output)
output = Dropout(0.5)(output)

# We project onto a single unit output layer, and squash it with a sigmoid:
output = Dense(1, activation='sigmoid')(output)
# output = Dense(1, activation='sigmoid')(output)


model = Model(inputs=[words_input], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc', evaluation.F1, evaluation.Recall, evaluation.Precision])

model.summary()

# RocAuc = RocAucEvaluation(validation_data=(X_dev,y_dev), interval=1)
# history = model.fit(X_train, y_train, batch_size=batch_size, epochs=15, validation_data=[X_dev, y_dev],callbacks=[RocAuc], verbose=2)

x_train, y_train, x_label, y_label = train_test_split(X_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(y_train, y_label), interval=1)
history = model.fit(x_train, x_label, batch_size=batch_size, epochs=nb_epoch, validation_data=[y_train, y_label],
                    callbacks=[RocAuc])


def my_plot(data_list, title, label_list, position):
    plt.subplot(position)
    plt.plot(data_list[0])
    plt.plot(data_list[1])
    plt.title(title)
    plt.ylabel(label_list[1])
    plt.xlabel(label_list[0])
    plt.legend(['Train', 'Test'], loc='upper left')


mean_acc = np.mean(history.history['acc'])
mean_val_acc = np.mean(history.history['val_acc'])

mean_loss = np.mean(history.history['acc'])
mean_val_loss = np.mean(history.history['val_acc'])

mean_F1 = np.mean(history.history['F1'])
mean_val_F1 = np.mean(history.history['val_F1'])

mean_Recall = np.mean(history.history['Recall'])
mean_val_Recall = np.mean(history.history['val_Recall'])

mean_Precision = np.mean(history.history['Precision'])
mean_val_Precision = np.mean(history.history['val_Precision'])

mean_score = np.mean(roc_list)

print('mean_acc:', mean_acc)
print('mean_val_acc: ', mean_val_acc)
print('mean_loss:', mean_loss)
print('mean_val_loss:', mean_val_loss)
print('mean_F1:', mean_F1)
print('mean_val_F1:', mean_val_F1)
print('mean_Recall:', mean_Recall)
print('mean_val_Recall:', mean_val_Recall)
print('mean_Precision:', mean_Precision)
print('mean_val_Precision:', mean_val_Precision)
print('mean_score:', mean_score)

my_plot([history.history['acc'], history.history['val_acc']], 'Model accuracy', ['Epoch', 'Accuracy'], 321)
my_plot([history.history['loss'], history.history['val_loss']], 'Model loss', ['Epoch', 'loss'], 322)
my_plot([history.history['F1'], history.history['val_F1']], 'Model F1', ['Epoch', 'F1'], 323)
my_plot([history.history['Recall'], history.history['val_Recall']], 'ModelRecall', ['Epoch', 'Recall'], 324)
my_plot([history.history['Precision'], history.history['val_Precision']], 'Model Precision', ['Epoch', 'Precision'],
        325)

plt.subplot(326)
plt.plot(roc_list)
plt.title('ROC_AUC')
plt.ylabel('ROC_AUC')
plt.xlabel('Epoch')
plt.legend(['ROC_AUC'], loc='upper left')

plt.show()
