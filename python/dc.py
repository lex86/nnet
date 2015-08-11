import os
import sys
import json
from multiprocessing import Queue, Event 
from multiprocessing.managers import BaseManager, ListProxy, EventProxy
from nnet import NNet

class Master:
    def __init__(self, cfg_path):
        if not os.path.exists(cfg_path):
            print 'Config file {} does not exist'.format(cfg_path)
            sys.exit(-1)

        with open(cfg_path) as cfg_file:
            self.cfg = json.load(cfg_file)

        self.nnet = NNet(cfg_path)
        self.data_queue = Queue()
        self.gradients_queue = Queue()
        self.event = Event()

        class NNetManager(BaseManager):
            pass

        NNetManager.register('get_cfg', callable=lambda: self.cfg, proxytype=DictProxy)
        NNetManager.register('get_data_queue', callable=lambda: self.data_queue)
        NNetManager.register('get_gradients', callable=lambda: self.gradients_queue)
        NNetManager.register('get_event', callable=lambda: self.event, proxtype=EventProxy)

        NNetManager.register('get_weights', callable=self.nnet.get_weights, proxtype=ListProxy)

        self.manager = NNetManager(address=(cfg['server_params.host'], cfg['server_params.port']),\
                authkey=cfg['server_params.key'])

        self.manager.start()

    def run(data):
        pass

class Worker:
    def __init__(self):
        class NNetManager(BaseManager):
            pass
        NNetManager.register('get_cfg')
        NNetManager.register('get_data_queue')
        NNetManager.register('get_gradients_queue')
        NNetManager.register('get_event')
        NNetManager.register('get_weights')

        self.manager = NNetManager(address=(cfg['server_params.host'], cfg['server_params.port']),\
                authkey=cfg['server_params.key'])

        self.manager.connect()

        self.cfg = self.manager.get_cfg()
        self.nnet = NNet(cfg_path)
        self.data_queue = self.manager.get_data_queue() 
        self.gradients_queue = self.manager.get_gradients_queue()
        self.event = self.manager.get_event()

    def run():
        pass
