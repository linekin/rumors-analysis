from time import time

import hdbscan

import jieba

jieba.set_dictionary('../data/dict.txt.big')
import pandas
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter


def load_text():
    data_path = '/Users/sean/g0v/rumors-db/opendata/articles.csv'
    df = pandas.read_csv(data_path)
    text = df['text'].tolist()
    print('documents', len(text))
    return text


def cluster(vectors):
    print('clustering...')
    t0 = time()
    # clusterer = hdbscan.HDBSCAN()
    # clusterer = DBSCAN()
    clusterer = AffinityPropagation(verbose=True)
    clusterer.fit(vectors)
    print('done in', time() - t0)

    clusterer_counts = Counter(clusterer.labels_)
    print(clusterer_counts)
    print('cluster number', max(clusterer_counts))

    return clusterer


def reduce_dimension():
    global t0
    print('reducing dimension...')
    t0 = time()
    X_pca = TruncatedSVD(1000).fit_transform(vectors)
    print('done in', time() - t0)
    cluster(X_pca)


if __name__ == '__main__':
    text = load_text()

    print('vectorizing')
    t0 = time()
    # vectorizer = CountVectorizer(
    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: jieba.cut(text),
        min_df=10
    )

    vectors = vectorizer.fit_transform(text)
    print('done in', time() - t0)
    print('vectors shape', vectors.shape)
    clusterer = cluster(vectors)





    # reduce_dimension()
