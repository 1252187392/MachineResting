#encoding:utf-8
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MR_Model.nn_models.word2vec import Word2Vec

class Word2vec_Extractor(object):

    def __init__(self,corpus_path=None):
        self.corpus_path = corpus_path
        self.model = Word2Vec(self.corpus_path)

    def fit(self):
        assert self.corpus_path != None
        self.model.preprocessing_data()
        lens = len(self.model.vocab2int)
        self.model.init_graph(lens)
        self.model.train()

    def word2int(self,word):
        return self.model.vocab2int(word)

    def int2word(self,int_word):
        return self.model.int2vocab(int_word)

    def word2vec(self,int_words):
        return self.model.get_vector(int_words)

    def load_mdoel(self,model_path):
        pass

    def save_model(self,model_path):
        pass

if __name__ == '__main__':
    pass