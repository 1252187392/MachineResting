#encoding:utf-8

from scipy import sparse
import numpy as np

class TfidfExtractor(object):
    #http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
    def __init__(self):
        self._idf = None

    def idf_extract(self,matrix,mode = 'train'):
        if mode == 'test':
            return self._idf
        samples,features = matrix.shape
        samples += 1
        features = features + 1
        _document_frequence = np.diff(matrix.indptr)
        idf = np.log(samples*1.0 / _document_frequence)
        self._idf = idf
        #print idf.shape
        return idf

    #def tf_extract(self,matrix):
    #    samples, features = matrix.shape
    #    _document_lens = matrix.sum(axis = 1).reshape((-1,1))
    #    tf = matrix / _document_lens
    #    return tf

    def tfidf_extract(self,matrix, mode = 'train'):
        tf = matrix
        idf = self.idf_extract(matrix)
        idf = sparse.diags(idf,0)
        tfidf = tf*idf
        print tfidf.toarray()
        print type(tfidf)

    def norm(self, matrix):
        matrix = matrix.dot()

if __name__ == '__main__':
    rows = [0,0,1,2,3,4,4,5,5]
    cols = [0,2,0,0,0,0,1,0,2]
    tf_data = [3,1,2,3,4,3,2,3,2]
    one_data = [1] * len(tf_data)
    matrix = sparse.csc_matrix((tf_data,(rows,cols)))
    print matrix.sum(axis=0)
    docs = matrix.shape[0] # 文档数
    words_sum = matrix.sum(axis=1)
    extractor = TfidfExtractor()
    extractor.tfidf_extract(matrix)
    x = [1,2,3]
    dd = sparse.diags(x,0)
    print dd.toarray()

'''
counts = [[3, 0, 1],
...           [2, 0, 0],
...           [3, 0, 0],
...           [4, 0, 0],
...           [3, 2, 0],
...           [3, 0, 2]]
...
array([[ 0.81940995,  0.        ,  0.57320793],
       [ 1.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ],
       [ 0.47330339,  0.88089948,  0.        ],
       [ 0.58149261,  0.        ,  0.81355169]])
'''
'''

[[3 0 1]
 [2 0 0]
 [3 0 0]
 [4 0 0]
 [3 2 0]
 [3 0 2]]
 
[[0.25      ]
 [0.5       ]
 [0.33333333]
 [0.25      ]
 [0.2       ]
 [0.2       ]]
 
'''