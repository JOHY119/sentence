import gzip
from pathlib import Path

import numpy as np
import pickle

import my_path

pklDir = Path(my_path.pkl_dir)


def generateParameters(negOrPos):
    path = Path(pklDir, negOrPos + '.pkl')
    with open(path, 'rb') as pkl:
        words = pickle.load(pkl)

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
    negWeights = generateParameters('neg')
    posWeights = generateParameters('pos')

    with gzip.open(Path(pklDir, 'emotion.pkl.gz'), 'wb') as f:
        pickle.dump({'neg': negWeights, 'pos': posWeights}, f)
