import sys
from ctypes import *
import numpy as np

nnetlib = np.ctypeslib.load_library('libnnet', '../lib/')

class NNet:
    def __init__(self, cfg_path):
        init = nnetlib.NNet_init
        self.nnet = init(cfg_path)
        if self.nnet == 0:
            sys.exit(-1)

        c_size = c_int()
        size = nnetlib.NNet_size
        size.argtypes = (c_void_p, POINTER(c_int))
        if size(self.nnet,
                byref(c_size)) < 0:
            sys.exit(-1)
        self.size = c_size.value

        c_num_iters = c_int()
        num_iters = nnetlib.NNet_num_iters
        num_iters.argtypes = (c_void_p, POINTER(c_int))
        if num_iters(self.nnet,
                byref(c_num_iters)) < 0:
            sys.exit(-1)
        self.num_iters = c_num_iters.value

        self.dims = np.zeros(self.size+1, dtype=np.int32)
        get_dims = nnetlib.NNet_get_dims
        get_dims.argtypes = (c_void_p,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.int32,
                    ndim = 1,
                    flags = 'C')
                )
        if get_dims(self.nnet,
                np.shape(self.dims)[0], self.dims) < 0:
            sys.exit(-1)

    def __del__(self):
        destroy = nnetlib.NNet_destroy
        destroy(self.nnet)
    def forward(self, data):
        forward = nnetlib.NNet_forward
        forward.argtypes = (c_void_p,
                c_int,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C')
                )
        if forward(self.nnet,
                np.shape(data)[0], np.shape(data)[1], data) < 0:
            sys.exit(-1)

    def backward(self, labels):
        backward = nnetlib.NNet_backward
        backward.argtypes = (c_void_p,
                c_int,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C')
                )
        if backward(self.nnet,
                np.shape(labels)[0],
                np.shape(labels)[1],
                labels) < 0:
            sys.exit(-1)
    def get_params(self):
        get_layer_params = nnetlib.NNet_get_layer_params
        get_layer_params.argtypes = (c_void_p,
                c_int,
                c_int,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C'),
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 1,
                    flags = 'C'))
        Ws = []
        bs = []
        for i in range(self.size):
            mat = np.zeros((self.dims[i], self.dims[i+1]), dtype=np.float64)
            vec = np.zeros(self.dims[i+1], dtype=np.float64)
            if get_layer_params(self.nnet,
                    i,
                    self.dims[i], self.dims[i+1], mat,
                    self.dims[i+1], vec) < 0:
                sys.exit(-1)
            Ws.append(mat)
            bs.append(vec)
        return Ws, bs
    def get_gradients(self):
        get_layer_gradients = nnetlib.NNet_get_layer_gradients
        get_layer_gradients.argtypes = (c_void_p,
                c_int,
                c_int,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C'),
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 1,
                    flags = 'C'))
        grad_Ws = []
        grad_bs = []
        for i in range(self.size):
            mat = np.zeros((self.dims[i], self.dims[i+1]), dtype=np.float64)
            vec = np.zeros(self.dims[i+1], dtype=np.float64)
            if get_layer_gradients(self.nnet,
                    i,
                    self.dims[i], self.dims[i+1], mat,
                    self.dims[i+1], vec) < 0:
                sys.exit(-1)
            grad_Ws.append(mat)
            grad_bs.append(vec)
        return grad_Ws, grad_bs
    def set_params(self, Ws, bs):
        set_layer_params = nnetlib.NNet_set_layer_params
        set_layer_params.argtypes = (c_void_p,
                c_int,
                c_int,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C'),
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 1,
                    flags = 'C'))
        for i in range(self.size):
            if set_layer_params(self.nnet,
                    i,
                    Ws[i].shape[0], Ws[i].shape[1], Ws[i],
                    bs[i].shape[0], bs[i]) < 0:
                sys.exit(-1)
    def update_params(self, grad_Ws, grad_bs):
        update_layer_params = nnetlib.NNet_update_layer_params
        update_layer_params.argtypes = (c_void_p,
                c_int,
                c_int,
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C'),
                c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 1,
                    flags = 'C'))
        for i in range(self.size):
            if update_layer_params(self.nnet,
                    i,
                    grad_Ws[i].shape[0], grad_Ws[i].shape[1], grad_Ws[i],
                    grad_bs[i].shape[0], grad_bs[i]) < 0:
                sys.exit(-1)
