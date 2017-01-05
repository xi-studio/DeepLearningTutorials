"""
    load_data()
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit
import gzip
import pickle

import numpy
import numpy as np
import cPickle
from scipy import misc
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
    def get_params(self,):
        return self.params

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input_now, input_mem, n_in, n_out, n_mem, W=None, W_m=None, b=None,
                 activation=T.tanh):
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if W_m is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_mem)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W_m = theano.shared(value=W_values, name='W_m', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.W_m = W_m
        self.b = b
        
        lin_output = T.dot(input_now, self.W) + T.sum(input_mem.T * self.W_m,axis=1) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.W_m, self.b]

        

# start-snippet-2
class MLP(object):
    def __init__(self, rng, input_now, input_mem, n_in, n_mem, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input_now=input_now,
            input_mem=input_mem,
            n_in=n_in,
            n_mem=n_mem,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

def sample(num = 100):
    res = np.append(np.random.np.random.random_sample(num)*6,np.random.np.random.random_sample(num)*(-6))
    np.random.shuffle(res)
    x_in = res.reshape((num,2))
   
    x_in = (x_in).astype(np.int8)
    y_out = (x_in[:,0] * x_in[:,1] > 0) * 1
      
    x_in = np.random.randn(num,100)
    y_out = (np.sum(x_in,axis=1)>0)*1
    y_out.astype(np.int8)

    return x_in,y_out

def load_data():
    x_in = np.load("train.npy")
    y_out = np.load("valid.npy")
    train_set = (x_in,y_out)
    x_in = np.load("train_1.npy")
    y_out = np.load("valid_1.npy")
    valid_set = (x_in,y_out) 
    
#    train_set = sample(1000)
#    valid_set = sample(1000)

    def shared_dataset(data_xy, borrow=True):
        data_x,data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    samples = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return samples


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             n_in=10, n_mem=4, batch_size=1,n_hidden=10):

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size


    print('... building the model')

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    x_mem = T.matrix('x_mem')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
   

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input_now=x,
        input_mem=x_mem,
        n_in=n_in,
        n_hidden=n_hidden,
        n_mem = 4,
        n_out=2
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            x_mem: valid_set_x[index * batch_size - n_mem:index * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            x_mem: train_set_x[index * batch_size - n_mem: index * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_mem,n_train_batches):
 
           # res = train_set_x[minibatch_index * batch_size - n_mem: (minibatch_index ) * batch_size ]
           # print(res.eval().shape) 
            minibatch_avg_cost = train_model(minibatch_index)
           # print(minibatch_avg_cost)


            # iteration number
            iter = (epoch - 1) * (n_train_batches) + minibatch_index 
             

            if (iter + 1) % (validation_frequency) == 0:
           
                # compute zero-one loss on validation set
                res = valid_set_x[index * batch_size - n_mem:index * batch_size],
                validation_losses = [validate_model(i) for i
                                     in range(n_mem,n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter


            if patience <= iter:
                print("No patience")
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    
if __name__ == '__main__':
    test_mlp()
