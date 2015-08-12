import os
import sys
import json
from multiprocessing import Queue, Event 
from multiprocessing.managers import BaseManager, ListProxy, EventProxy
from nnet import NNet
import numpy as np

class Master:
    def __init__(self, cfg_path):
        if not os.path.exists(cfg_path):
            print 'Config file {} does not exist'.format(cfg_path)
            sys.exit(-1)

        with open(cfg_path) as cfg_file:
            self.cfg = json.load(cfg_file)

        self.nnet = NNet(cfg_path)

        weights_list, biases_list = self.nnet.get_params()
        params = {
                'Ws': weights_list,
                'bs': biases_list
                }
        data_queue = Queue()
        result_queue = Queue()
        event = Event()
        sync_dict = {
                'is_data_queue_completed': False,
                'is_iteration_compleded' : False
                }

        class NNetManager(BaseManager):
            pass

        NNetManager.register('get_data_queue', callable=lambda: data_queue)
        NNetManager.register('get_result_queue', callable=lambda: result_queue)
        NNetManager.register('get_event', callable=lambda: event, proxtype=EventProxy)
        NNetManager.register('get_params', callable=lambda: params, proxtype=DictProxy)
        NNetManager.register('get_sync_dict', callable=lambda: sync_dict, proxtype=DictProxy)

        self.manager = NNetManager(address=(cfg['server_params.host'], cfg['server_params.port']),\
                authkey=cfg['server_params.key'])

        self.manager.start()

        self.data_queue = self.manager.get_data_queue()
        self.result_queue = self.manager.get_result_queue()
        self.event = self.manager.get_event()
        self.params = self.manager.get_params()
        self.sync_dict = self.manager.get_sync_dict()
        self.grads = {'grad_Ws': [],'grad_bs': []}

        for idx in xrange(self.nnet.size):
            self.grads['grad_Ws'].append(np.zeros((self.nnet.dims[idx], self.nnet.dims[idx+1]),
                np.dtype=np.float64))
            self.grads['grad_bs'].append(np.zeros(self.nnet.dims[idx+1],
                np.dtype=np.float64))

    def start(data, labels):

        print 'Neural Network training...'

        self.splited_data = np.array_split(data, 10)
        self.splited_labels = np.array_split(labels, 10)
        self.splited_len = len(splited_data) 

        for iteration in xrange(nnet_iter):

            for idx in xrange(self.nnet.size):
                self.grads['grad_Ws'][idx].fill(0.0)
                self.grads['grad_bs'][idx].fill(0.0)

            print 'Iteration {}'.format(iteration)

            self.sync_dict["is_data_queue_completed"] = False 

            for chunk in xrange(self.splited_len):
                data_queue.put((self.splited_data[chunk], self.splited_labels[chunk]))

            self.sinc_dict["is_data_queue_completed"] = True

            self.event.clear()

            for chunk in xrange(self.splited_len):
                result = self.result_queue.get()
                for idx in xrange(self.nnet.size):
                    grads['grad_Ws'][idx] = np.add(grads['grad_Ws'][idx], result['grad_Ws'][idx])
                    grads['grad_bs'][idx] = np.add(grads['grad_bs'][idx], result['grad_Ws'][idx])

            print 'Updating neural network parameters...'

            self.nnet.update_params(grads['grad_Ws'], grads['grad_bs'])

            weights_list, biases_list = self.nnet.get_params()
            self.params['Ws'] = weights_list
            self.params['bs'] = biases_list

            if iteration == nnet_iter - 1:
                self.sync_dict["is_iteration_compleded"] = True 

            self.event.set()


class Worker:
    def __init__(self, cfg_path):
        class NNetManager(BaseManager):
            pass
        NNetManager.register('get_data_queue')
        NNetManager.register('get_result_queue')
        NNetManager.register('get_event')
        NNetManager.register('get_params')

        self.manager = NNetManager(address=(cfg['server_params.host'], cfg['server_params.port']),\
                authkey=cfg['server_params.key'])

        self.manager.connect()

        self.nnet = NNet(cfg_path)
        self.data_queue = self.manager.get_data_queue() 
        self.result_queue = self.manager.get_result_queue()
        self.event = self.manager.get_event()
        self.params = self.manager.get_params()

    def run():

        print 'Neural Network training...'

        while not self.sync_dict['is_iteration_compleded']:

            self.nnet.set_params(self.params['Ws'], self.params['bs'])
            
            while not self.sync_dict['is_data_queue_completed']:
                work()

            self.event.wait()

    def work():
        try:
            data = self.data_queue.get(True, 1)
            self.nnet.forward(data[0])
            self.nnet.backward(data[1])
            grad_Ws, grad_bs = self.nnet.get_gradients()
            result = {'grad_Ws': grad_Ws, 'grad_bs': grad_bs}
            self.result_queue.put(result)
        except: Queue.Empty:
            return

