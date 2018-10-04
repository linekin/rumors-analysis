import pyLDAvis

from models.cluster import load_text
from gensim import corpora,models
import jieba

jieba.set_dictionary('../data/dict.txt.big')
import os

texts = load_text()
texts = (jieba.cut(line) for line in texts)

dictionary_file = '../data/text.dict'
if os.path.exists(dictionary_file):
    dictionary = corpora.Dictionary.load(dictionary_file)
else:
    dictionary = corpora.Dictionary(texts)
    dictionary.save('../data/text.dict')


corpus_file = '../data/text.mm'
if os.path.exists(corpus_file):
    corpus = corpora.MmCorpus(corpus_file)
else:
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(corpus_file, corpus)


model = models.LdaModel(corpus)

data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.save_html(data, 'lda.html')