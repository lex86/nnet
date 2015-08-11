import ctypes
import numpy as np

nnetlib = np.ctypeslib.load_library('libnnet', '../lib/')

class NNet:
    def __init__(self, cfg_path):
        init = nnetlib.NNet_init
        self.nnet = init(cfg_path)
    def __del__(self):
        destroy = nnetlib.NNet_destroy
        destroy(self.nnet)
    def forward(self, data):
        forward = nnetlib.NNet_forward
        forward.argtypes = (ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C')
                )
        return forward(self.nnet, np.shape(data)[0], np.shape(data)[1], data)
    def backward(self, labels):
        backward = nnetlib.NNet_backward
        backward.argtypes = (ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                np.ctypeslib.ndpointer(
                    dtype = np.float64,
                    ndim = 2,
                    flags = 'C')
                )
        return backward(self.nnet, np.shape(labels)[0], np.shape(labels)[1], labels)

    def size(self):
        result = 0
        if self.size == -1:
            size = nnetlib.NNet_size
            result = size(self.nnet, byref(self.size))
        return result
