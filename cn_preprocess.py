import gzip
import logging
import pickle as pkl

# We download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
from pathlib import Path

import jieba
import numpy as np

import my_path

embeddingsPath = Path(my_path.embedding_path)
stopWordsPath = Path(my_path.stopping_path)

# outputFilePath = 'pkl/ShareData.pkl.gz'
outputFilePath = Path(my_path.pkl_dir, 'data.pkl.gz')

# Train, Dev, and Test files
# folder = 'D:/Data/sentiment/shopping/'
folder = Path(my_path.data_dir)
files = [Path(folder, 'cnews.train.txt'), Path(folder, 'cnews.dev.txt'), Path(folder, 'cnews.test.txt')]
# files = [folder + 'train.txt', folder + 'dev.txt', folder + 'test.txt']
# categories = ['neg', 'pos']
categories = ['体育','娱乐','家居','房产','教育','时尚','时政','游戏','科技','财经']


def createMatrices(sentences, word2Idx, stop_words):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    xMatrix = []
    unknownWordCount = 0
    wordCount = 0
    stopCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        sentenceWordIdx = []

        for word in sentence:
            wordCount += 1

            if word in stop_words:
                stopCount += 1
                continue

            if word in word2Idx:
                wordIdx = word2Idx[word]
            # elif word.lower() in word2Idx:
            #     wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1

            sentenceWordIdx.append(wordIdx)

        xMatrix.append(sentenceWordIdx)

    print("Unknown tokens: %.2f%%" % (unknownWordCount / (float(wordCount)) * 100))
    print("Stop tokens: %.2f%%" % (stopCount / (float(wordCount)) * 100))

    return xMatrix



def get_label(word):
    return categories.index(word)


def get_words_list(sentence):
    words = list(jieba.cut(sentence))

    return words


def get_stop_word_dict(path):
    with open(path, encoding='utf-8') as f:
        dic = {}
        for line in f:
            word = line.strip()
            dic[word] = True
        dic[' '] = True
        return dic


def readFile(filepath):
    sentences = []
    labels = []
    index = 0
    for line in open(filepath, encoding='utf-8'):
        index += 1
        try:
            label, content = line.strip().split('\t')
            if content:
                words = get_words_list(content)
                sentences.append(words)
                if index % 1000 == 0:
                    logging.warning('分词中，' + str(index) + '行')
                label = get_label(label)
                labels.append(label)
        except Exception as e:
            print(e)
            logging.warning('行' + str(index) + '读取出错， 继续')
            pass

    print(filepath, len(sentences), "sentences")

    return sentences, labels


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

if __name__ == '__main__':

    # outputFilePath = 'pkl/ShareData.pkl.gz'

    trainDataset = readFile(files[0])
    devDataset = readFile(files[1])
    testDataset = readFile(files[2])

    # :: Compute which words are needed for the train/dev/test set ::
    words = {}
    for sentences, labels in [trainDataset, devDataset, testDataset]:
        for sentence in sentences:
            for token in sentence:
                words[token] = True

    stop_word_dict = get_stop_word_dict(stopWordsPath)
    # :: Read in word embeddings ::
    word2Idx = {}
    wordEmbeddings = []

    # :: Load the pre-trained embeddings file ::
    fEmbeddings = open(embeddingsPath, encoding="utf8")

    print("Load pre-trained embeddings file")
    for line in fEmbeddings:
        split = line.strip().split(" ")
        if len(split) < 200:
            continue
        word = split[0]

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        if word in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[word] = len(word2Idx)

    wordEmbeddings = np.array(wordEmbeddings)

    print("Embeddings shape: ", wordEmbeddings.shape)
    print("Len words: ", len(words))

    # :: Create matrices ::
    train_matrix = createMatrices(trainDataset[0], word2Idx, stop_word_dict)
    dev_matrix = createMatrices(devDataset[0], word2Idx, stop_word_dict)
    test_matrix = createMatrices(testDataset[0], word2Idx, stop_word_dict)


    data = {
        'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
        'train': {'sentences': train_matrix, 'labels': trainDataset[1]},
        'dev': {'sentences': dev_matrix, 'labels': devDataset[1]},
        'test': {'sentences': test_matrix, 'labels': testDataset[1]}
    }

    f = gzip.open(outputFilePath, 'wb')
    pkl.dump(data, f)
    f.close()

    print("Data stored in pkl folder")
