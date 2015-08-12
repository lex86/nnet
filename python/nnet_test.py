#!/usr/bin/python

import numpy as np
from nnet import NNet, nnetlib

if __name__ == '__main__':
    nn = NNet('config.json')
    data = np.random.rand(5, 4)
    print data
    nn.forward(data)
    labels = np.random.rand(5, 2)
    nn.backward(labels)
    print 'NNet size = ', nn.size
    print 'NNet dims = ', nn.dims
    print 'NNet num_iters = ', nn.num_iters
    Ws, bs = nn.get_params()
    grad_Ws, grad_bs = nn.get_gradients()
    print Ws[0]
    print bs[0]
    Ws[0] = Ws[0]+100.0
    bs[0] = bs[0]+10.0
    nn.set_params(Ws, bs)
    Ws1, bs1 = nn.get_params()
    print Ws1[0]
    print bs1[0]
    nn.update_params(Ws1, bs1)
    Ws1, bs1 = nn.get_params()
    print Ws1[0]
    print bs1[0]




