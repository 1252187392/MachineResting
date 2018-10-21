#encoding:utf-8

import tensorflow as tf
from collections import Counter
import numpy as np
import os
import itertools
import time
import pickle
import random
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #cpu mode
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def read_vectors(path):
    word_cnt,dim = 0,0
    vectors = []
    iw = []
    print path
    with open(path) as f:
        first_line = True
        cnt = 0
        for line in f:
            if dim == 0:
                word_cnt = int(line.rstrip().split()[0])
                dim = int(line.rstrip().split()[1])
                continue
            cnt += 1
            tokens = line.rstrip().split(' ')
            vectors.append([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if cnt % 10000 == 0:
                print 'finish {}'.format(cnt*1.0 / word_cnt)
    wi = {w:i for i,w in enumerate(iw)}
    return np.array(vectors,dtype=np.float32), iw, wi, dim

class Word2Vec(object):
    '''
    根据指定的语料数据训练word2vec模型
    语料格式  每一行为一个doc
    '''
    
    def __init__(self,data=None,config=None):
        '''
        :param data: 语料序列或迭代器
        '''
        self.corpus = data
        self.graph = tf.Graph()
        if config == None:
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth=True
        #self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session(graph=self.graph,config=config)
        self.word_size = 0
        self.vocab2int = {}
        self.int2vocab = []
        self.iterations = 0
        self.dim = -1
        self.pretrained_vector = None

    def load_word2vec_file(self,filename):
        vectors, iw, wi, dim = read_vectors(filename)
        self.vocab2int = wi
        self.int2vocab = iw
        self.word_size = len(iw)
        self.dim = dim
        self.pretrained_vector = vectors
        self.drop_proba = None
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
        self.corpus,corpus = itertools.tee(self.corpus)
        for docs in corpus:
            for w in docs:
                w = str(w)
                if w not in self.vocab2int:
                    self.vocab2int[w] = self.word_size
                    self.int2vocab.append(w)
                    self.word_size += 1
            word_total += len(docs)
            int_doc = [self.vocab2int[w] for w in docs]
            counters.update(int_doc)
            train_texts.append(int_doc)
            
        word_total *= 1.0
        t = 1e-5
        threshold = 0.8
        word_freq = {k:v/word_total for k,v in counters.iteritems()} #词频率
        self.drop_proba = {k:max(1 - np.sqrt(t / v),0)for k,v in word_freq.iteritems()} #丢弃的概率 大于该值保留

    def init_graph(self,n_sampled=100,embedding_size=300):
        '''
        定义graph
        :param vocab_size: 词表大小
        :param n_sampled: 负采样时用到的类别数
        :param embedding_size: 词向量的列数
        :return:
        '''
        if self.dim != -1:
            pretrained_size = self.pretrained_vector.shape[0]
        else:
            self.dim = embedding_size
            pretrained_size = 0
        
        print '总词表大小:',self.word_size
        print '预训练好的词表大小',pretrained_size,'新词数量',self.word_size - pretrained_size
        with self.graph.as_default():
            #相关参数的初始化
            with tf.variable_scope("embed"):
                if pretrained_size > 0:
                    embedding_part_1 = tf.Variable(self.pretrained_vector)
                    embedding_part_2 = tf.Variable(tf.random_uniform([self.word_size - pretrained_size ,self.dim],
                                                          -1,1),name='t2')
                    embedding = tf.concat([embedding_part_1,embedding_part_2],0,name='vocab_embed')
                else:
                    embedding = tf.Variable(tf.random_uniform([self.word_size,embedding_size],
                                                      -1,1),name='vocab_embed')
                weights = tf.Variable(tf.truncated_normal([self.word_size,embedding_size]))
                biases = tf.Variable(tf.zeros(self.word_size))
            with tf.name_scope("input_layer") as scope:
                input_x = tf.placeholder(tf.int32,shape=[None],name='input_x')
                input_y = tf.placeholder(tf.int32,shape=[None,None],name='input_y')
            with tf.name_scope("embedding_layer"):
                embedding_lookup = tf.nn.embedding_lookup(embedding, input_x,name='embedding_lookup')
            with tf.name_scope("output_layer"):
                loss = tf.nn.sampled_softmax_loss(weights,biases,input_y,embedding_lookup,
                                                  num_sampled=n_sampled,num_classes=self.word_size,name='loss')
            with tf.name_scope("cost"):
                cost = tf.reduce_mean(loss,name='cost')
                opt = tf.train.AdamOptimizer().minimize(cost,name='adam')

    def train(self, epoches = 10, batch_size = 500, window_size = 5,save_path = None, model_save_iter=5000, board_logdir = None):
        assert self.graph != None
        with self.graph.as_default():
            initer = tf.global_variables_initializer()
            saver = tf.train.Saver()

        with self.sess.as_default():
            initer.run()
            cost = self.graph.get_tensor_by_name('cost/cost:0')
            opt = self.graph.get_operation_by_name('cost/adam')
            cost_scalar = tf.summary.scalar('mean_cost',cost)
            if board_logdir:
                writer = tf.summary.FileWriter(board_logdir, self.graph)
            tot_loss = 0
            t1 = time.time()
            for epoch in range(epoches):
                for x, y in self.batch_generator(batch_size, window_size):
                    feed_dict={'input_layer/input_x:0':x,
                                'input_layer/input_y:0':y}                   
                    loss, _= self.sess.run([cost, opt],feed_dict=feed_dict)
                    tot_loss += loss
                    self.iterations += 1
                    if self.iterations % 100 == 0:
                        t2 = time.time()
                        print 'Epoch:{}\titerations:{}\taver_loss:{}\ttime:{}'.format(epoch,self.iterations,tot_loss/100,t2-t1)
                        tot_loss = 0
                        t1 = t2
                    if board_logdir and self.iterations % 1000 == 0:
                        rs = self.sess.run(cost_scalar,feed_dict=feed_dict)
                        writer.add_summary(rs,self.iterations/1000)
                    if save_path and self.iterations % save_iter == 0:
                        saver.save(self.sess, save_path, global_step = self.iterations)
            print 'Epoch:{}\titerations:{}\taver_loss:{}\ttime:{}'.format(epoch,self.iterations,tot_loss/100,time.time()-t1)
            if save_path:
                _path = saver.save(self.sess, save_path, global_step = self.iterations)
                print 'save_path',_path
            if board_logdir:
                writer.close()
                
    def batch_generator(self,batch_size, window_size):
        '''
        :param batch_size:
        :param window_size:
        :return:
        '''
        #rubbish = set([])
        batch_x, batch_y = [],[]
        self.corpus,corpus = itertools.tee(self.corpus)
        for doc in corpus:
            int_text = [self.vocab2int.get(w,-1) for w in doc]
            for key_index, int_word in enumerate(int_text):
                rand_x = random.uniform(0,1)
                if rand_x + 0.05 < self.drop_proba.get(int_word,0):
                    #rubbish.add(str(self.int2vocab[int_word])+str(self.drop_proba.get(int_word,0)))
                    continue
                left = max(0,key_index - window_size)
                right = min(len(int_text),key_index+window_size+1)
                for i in range(left, right):
                    batch_x.append(int_word)
                    batch_y.append(int_text[i])
                    if len(batch_x) == batch_size:
                        yield batch_x,np.array(batch_y).reshape(-1,1)
                        batch_x,batch_y = [],[]
        
    def restore(self,meta_path,model_dir,int2vocab_file):
        self.load_graph(meta_path, model_dir)
        self.int2vocab = pickle.load(open(int2vocab_file))
        self.vocab2int = {w:i for i,w in enumerate(self.int2vocab)}
    
    def load_graph(self,meta_path,model_dir):
        '''
        :param meta_path:
        :param model_dir:
        :return:
        Todo:test
        '''
        with self.graph.as_default():
            loader = tf.train.import_meta_graph(meta_path)
            print type(loader)
            loader.restore(self.sess,tf.train.latest_checkpoint(model_dir))
        
    def dump(self,save_path):
        '''
        model save
        '''
        if save_path[-1] != '/':
            save_path+='/'
        #joblib.dump(self.int2vocab,save_path+'int2vocab')
        pickle.dump(self.int2vocab, open(save_path+'int2vocab.json','w'))
        with self.graph.as_default():
            saver = tf.train.Saver()
        with self.sess.as_default():
            saver.save(self.sess, save_path+'word2vec', global_step = self.iterations)

    def word2int(self, words):
        return [self.vocab2int.get(word, -1) for word in words]

    def int2word(self, indexes):
        return [self.int2vocab[ind] if ind < self.word_size else None for ind in indexes]

    def get_vector(self,int_words):
        assert self.graph != None
        with self.sess.as_default():
            embedding_lookup = self.graph.get_tensor_by_name('embedding_layer/embedding_lookup:0')
            vec = self.sess.run(embedding_lookup,
                           feed_dict={'input_layer/input_x:0':int_words})
        return vec
    
    def close_session(self):
        self.sess.close()

    def export(self,filename):
        with open(filename,'w') as fout:
            int_words = [i for i in range(self.word_size)]
            vecs = self.get_vector(int_words)
            words = self.int2word(int_words)            
            for w,v in zip(words,vecs):
                print >> fout, '{}\t{}'.format(w,' '.join([str(_) for _ in v]))
