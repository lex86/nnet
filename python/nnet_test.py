#!/usr/bin/python

import numpy as np
from nnet import NNet, nnetlib


if __name__ == '__main__':
    nn = NNet('config.json')
    data = np.random.rand(10, 768)
    nn.forward(data)
    labels = np.random.rand(10, 2)
    nn.backward(labels)

