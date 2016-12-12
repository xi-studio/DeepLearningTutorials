from __future__ import print_function
from collections import OrderedDict
import cPickle
import copy
import gzip
import os
import urllib
import random
import stat
import subprocess
import sys
import timeit

import numpy as np
import numpy

import theano
from theano import tensor as T

import sys
sys.setrecursionlimit(1500)


# utils functions
def rollarray(ls):
    ls = np.append(ls,0)
    base = []
    for x in range(7):
        base.append(np.roll(ls,-x))
    res = np.array(base).reshape(7,len(ls))
    res = res.T
    lex = res[:-7]
    pred = res[1:-6] 
    return lex,pred

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


# start-snippet-1
def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out
# end-snippet-1


# data loading functions
def load_data(filename):
    f = gzip.open(filename, 'rb')
    squence = cPickle.load(f)
    y = []
    x_set = np.ones((1,7))
    y_set = np.ones((1,7))
    for data in squence:
	res_x,res_y = rollarray(data)
	x_set = np.append(x_set,res_x,axis=0)
	y_set = np.append(y_set,res_y,axis=0)
    
    x_set = x_set.astype(np.int16)
    y_set = y_set.astype(np.int16)
    index = int(np.ceil(x_set.shape[0]*0.8))
    train_set, test_set = (x_set[1:index],y_set[1:index]),(x_set[index:],y_set[index:]) 
    return train_set, test_set

def load_data_new(filename):
    f = gzip.open(filename, 'rb')
    squence = cPickle.load(f)
    x_set = np.ones((1,7))
    y_set = np.ones(1)
    for data in squence:
        res = contextwin(data,7)
        x_set = np.append(x_set,res[:-1],axis=0)
        y_set = np.append(y_set,data[1:])
    
    x_set = x_set.astype(np.int16)
    y_set = y_set.astype(np.int16)
    index = int(np.ceil(x_set.shape[0]*0.8))
    train_set, test_set = (x_set[1:index],y_set[1:index]),(x_set[index:],y_set[index:]) 
    train_set, test_set = (x_set[1:index],y_set[1:index]),(x_set[index:],y_set[index:]) 
    return train_set, test_set





# start-snippet-2
class RNNSLU(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.emb, self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]
        # end-snippet-2
        # as many columns as context window size
        # as many lines as words in the sentence
        # start-snippet-3
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels
        # end-snippet-3 start-snippet-4

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        # end-snippet-4

        # cost and gradients and learning rate
        # start-snippet-5
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))
        # end-snippet-5

        # theano functions to compile
        # start-snippet-6
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        # end-snippet-6 start-snippet-7
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        # end-snippet-7

    def train(self, x, y, learning_rate):
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), x))
        labels = y

        self.sentence_train(words, labels, learning_rate)
        self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))


def main(param=None):
    if not param:
        param = {
            'data': 'atis',
            'lr': 0.0970806646812754,
            'verbose': 1,
            'decay': True,
            # decay on the learning rate if improvement stops
            'win': 7,
            # number of words in the context window
            'nhidden': 200,
            # number of hidden units
            'seed': 345,
            'emb_dimension': 8,
            # dimension of word embedding
            'nepochs': 60,
            # 60 is recommended
            'savemodel': True,
	    'batch_size':60}
    print(param)

    folder = './rnnmidi/'
    # load the dataset
    train_set, test_set = load_data_new("../data/line_midi/base_squence.pkl.gz")

    train_lex, train_y = train_set
    test_lex,  test_y = test_set

    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    batch_size = param['batch_size']
    n_train_batches = train_lex.shape[0]//batch_size
    n_test_batches = test_lex.shape[0]//batch_size

    rnn = RNNSLU(nh=param['nhidden'],
                 nc=125,
                 ne=256,
                 de=param['emb_dimension'],
                 cs=param['win'])

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

        # shuffle
        shuffle([train_lex, train_y], param['seed'])

        param['ce'] = e
        tic = timeit.default_timer()

        for index in range(n_train_batches):
	    x = train_lex[index * batch_size:(index + 1) * batch_size]
	    y = train_y[index * batch_size:(index + 1) * batch_size]

            rnn.train(x, y, param['clr'])
            print('[learning] epoch %i >> %2.2f%%' % (
                e, (index + 1) * 100. / n_train_batches), end=' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), end='')
            sys.stdout.flush()

        #ronn.classify(numpy.asarray(contextwin(x, param['win'])).astype('int32'))
        res = []
        for index in range(n_test_batches):
	    x = test_lex[index * batch_size:(index + 1) * batch_size]
	    y = test_y[index * batch_size:(index + 1) * batch_size]

	    pred = rnn.classify(x).astype('int32')
	    error = numpy.mean(numpy.equal(pred,y))
	    res.append(error)

            print('[testing] epoch %i >> %2.2f%%' % (
                e, (index + 1) * 100. / n_test_batches), end=' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), end='')
            sys.stdout.flush()

	res_test = numpy.mean(numpy.array(res))
	

        if res_test > best_f1:

            if param['savemodel']:
                rnn.save(folder)

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_test

            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'test', res_test)

            param['tf1'] = res_test
            param['be'] = e

        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['tf1'])

def predict(param=None):
    if not param:
        param = {
            'lr': 0.0970806646812754,
            'verbose': 1,
            'decay': True,
            'win': 7,
            'nhidden': 200,
            'seed': 345,
            'emb_dimension': 8,
            'nepochs': 60,
            'savemodel': True}
    print(param)

    rnn = RNNSLU(nh=param['nhidden'],
                 nc=125,
                 ne=256,
                 de=param['emb_dimension'],
                 cs=param['win'])
    folder = './rnnmidi/'
    print("loading")
    rnn.load(folder)
    train_set, test_set = load_data_new("../data/line_midi/base_squence.pkl.gz")
    lex,cl = train_set 

    steps = 100
    x = lex[5]
    res = []
    for num in range(steps):
        y = rnn.classify([x]).astype('int32')
	print('pred',y)
	print('fact',cl[num])
	x = np.append(x,y[-1])
	x = np.delete(x,0)
	#print(x)
	res.append(y[-1])
	#print(num)
    print('result',res)
    #print('fact',lex[:10])


if __name__ == '__main__':
   main()
   #predict()
