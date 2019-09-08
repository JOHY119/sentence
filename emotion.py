"""
取embedding中积极情感和消极情感最强烈的词各2048个
"""
import gzip
import logging
import pickle
from pathlib import Path
import numpy as np
import my_path

logging.basicConfig(level=logging.DEBUG)

data_dir = Path(my_path.data_dir)
pkl_dir = Path(my_path.pkl_dir)
embedding_dir = Path(my_path.embedding_dir)

embedding = Path(embedding_dir)
e_list_path = Path(my_path.e_list_path)

# 需生成的情感词数量 neg pos数量相同
nb_words = 2048


def get_embedding_dict(path):
    logging.debug('生成embedding字典中')

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        # no = int(lines[0].split(' ')[0])
        # shape = int(lines[0].split(' ')[1])
        embedding_list = [line.strip().split(' ') for line in lines[1:]]
        embedding_dic = dict(zip([x[0] for x in embedding_list], [x[1:] for x in embedding_list]))

    return embedding_dic


def get_emotion_words_list(path):
    logging.debug('生成所有感情词列表中')

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        emotion_list = [line.strip().split(' ') for line in lines]
        return emotion_list


def find_words(embedding_dict, emotion_list: list, neg_or_pos, n=2048):
    logging.debug('生成列表，词性：' + neg_or_pos)
    word_list = []

    def judge(np, num):

        return (True if num < 0 else False) if np == 'neg' else (True if num > 0 else False)

    if neg_or_pos == 'pos':
        emotion_list.reverse()

    # 词典已经排好序 故不重新排序
    index = 0
    for word in emotion_list:
        if word[0] in embedding_dict:
            if index == n:
                break
            if not judge(neg_or_pos, float(word[1])):
                logging.warning(str(index) + ' ' + neg_or_pos + '词不够，' + str(word))
            index += 1
            c_word = word.copy()
            c_word[1] = float(c_word[1])
            e_l = embedding_dict[word[0]].copy()
            e_l = [float(n) for n in e_l]
            c_word.append(e_l)
            word_list.append(c_word)

    return word_list


def generateParameters(words):
    # path = Path(pklDir, negOrPos + '.pkl')
    # with open(path, 'rb') as pkl:
    #     words = pickle.load(pkl)

    embeddings = [l[2] for l in words]
    weights = np.array(embeddings)
    weights = weights.T
    word_filter_weights = np.expand_dims(weights, axis=0)

    bias_weights = np.zeros(word_filter_weights.shape[2])  # or random ???
    # bias_weights = np.random.uniform(low=-0.05, high=0.05, size=word_filter_weights.shape[2])   # adjust???
    # 将两个矩阵放在同一个list中，作为返回值
    cnn_wordfilter_weights = [word_filter_weights, bias_weights]
    return cnn_wordfilter_weights


if __name__ == '__main__':
    embedding_dict = get_embedding_dict(embedding)
    word_list = get_emotion_words_list(e_list_path)

    neg_list = find_words(embedding_dict, word_list, 'neg', nb_words)
    pos_list = find_words(embedding_dict, word_list, 'pos', nb_words)

    negWeights = generateParameters(neg_list)
    posWeights = generateParameters(pos_list)

    print('neg shape: ' + str(negWeights[0].shape))
    print('pos shape: ' + str(posWeights[0].shape))

    with gzip.open(Path(pkl_dir, 'emotion.pkl.gz'), 'wb') as f:
        pickle.dump({'neg': negWeights, 'pos': posWeights}, f)
