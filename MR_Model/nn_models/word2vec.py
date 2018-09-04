#encoding:utf-8

import tensorflow as tf
from collections import Counter
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Word2Vec(object):
    '''
    根据指定的语料数据训练word2vec模型
    语料格式  每一行为一个doc
    '''

    def __init__(self,corpus):
        '''
        :param corpus: 语料路径
        '''
        self.corpus = corpus
        self.model = None
        self.vocab2int = {}
        self.int2vocab = {}

    def preprocessing_data(self):
        '''
        数据预处理
        1.词编号
        2.词频统计
        3.词筛选
        :return:
        '''
        counters = Counter()
        word_total = 0
        train_texts = []
        with open(self.corpus) as fin:
            index = 1
            for line in fin:
                line = line.strip().split(' ')
                for w in line:
                    if w not in self.vocab2int:
                        self.vocab2int[w] = index
                        self.int2vocab[index] = w
                        index += 1
                word_total += len(line)
                int_sequence = [self.vocab2int[w] for w in line] #one text
                counters.update(int_sequence)
                train_texts.append(int_sequence)
        word_total *= 1.0
        t = 1e-5
        threshold = 0.8
        word_freq = {k:v/word_total for k,v in counters.iteritems()} #词频率
        #word_proba = {k:1 - np.sqrt(t / v) for k,v in word_freq.iteritems()}

    def init_graph(self,vocab_size,n_sampled=100,embedding_size=300):
        '''
        初始化模型
        :param vocab_size: 词表大小
        :param n_sampled: 负采样时用到的类别数
        :param embedding_size: 词向量的列数
        :return:
        '''
        self.model = tf.Graph()

        with self.model.as_default():
            input_x = tf.placeholder(tf.int32,shape=[None],name='input_x')
            input_y = tf.placeholder(tf.int32,shape=[None,None],name='input_y')
            #定义embedding layer

            embedding = tf.Variable(tf.random_uniform([vocab_size,embedding_size],
                                                      -1,1),name='embed')
            embedding_lookup = tf.nn.embedding_lookup(embedding, input_x,name='embedding_lookup')
            #定义softmax layer
            weights = tf.Variable(tf.truncated_normal([vocab_size,embedding_size]))
            biases = tf.Variable(tf.zeros(vocab_size))
            loss = tf.nn.sampled_softmax_loss(weights,biases,input_y,embedding_lookup,
                                              num_sampled=n_sampled,num_classes=vocab_size)
            cost = tf.reduce_mean(loss,name='cost')
            opt = tf.train.AdamOptimizer().minimize(cost,name='opt')


    def train(self, epoches = 10, batch_size = 500, window_size = 5,save_path = None, save_iter=5000):
        assert self.model != None
        with self.model.as_default():
            initer = tf.global_variables_initializer()
            saver = tf.train.Saver()
        with tf.Session(graph=self.model) as sess:
            print 1
            initer.run()
            cost = self.model.get_tensor_by_name('cost:0')
            print cost
            opt = self.model.get_operation_by_name('opt')
            tot_loss = 0
            iterations = 0
            for epoch in range(epoches):
                for x, y in self.batch_generator(batch_size, window_size):
                    loss, _= sess.run([cost, opt],feed_dict={'input_x:0':x,
                                                            'input_y:0':y})
                    tot_loss += loss
                    iterations += 1
                    if iterations % 100 == 0:
                        print 'Epoch:{}\titerations:{}\t,aver_loss:{}\t'.format(epoch,iterations,tot_loss/100)
                        tot_loss = 0
                    if save_path and iterations % save_iter == 0:
                        saver.save(sess, save_path, global_step = iterations)

    def batch_generator(self,batch_size, window_size):
        '''
        :param batch_size:
        :param window_size:
        :return:
        '''
        with open(self.corpus) as fin:
            batch_x, batch_y = [], []
            for text in fin:
                text = text.strip().split(' ')
                int_text = [self.vocab2int.get(w,-1) for w in text]
                for key_index, key_word in enumerate(int_text):
                    left = max(0,key_index - window_size)
                    right = min(len(int_text),key_index+window_size+1)
                    for i in range(left, right):
                        batch_x.append(key_word)
                        batch_y.append(int_text[i])
                        if len(batch_x) == batch_size:
                            yield batch_x,np.array(batch_y).reshape(-1,1)
                            batch_x,batch_y = [],[]

    def load_model(self,meta_path,model_dir):
        '''
        :param meta_path:
        :param model_dir:
        :return:
        '''
        with tf.Session() as sess:
            loader = tf.train.import_meta_graph(meta_path)
            loader.restore(sess,tf.train.latest_checkpoint(model_dir))
            self.model =sess.graph

    def word2int(self, word):
        return self.vocab2int.get(word, -1)

    def int2word(self, index):
        return self.int2vocab.get(index, -1)

    def get_vector(self,int_words):
        assert self.model != None
        with tf.Session(graph=self.model) as sess:
            embedding_lookup = self.model.get_tensor_by_name('embedding_lookup:0')
            vec = sess.run([embedding_lookup],
                           feed_dict={'input_x:0':int_words})
            return vec