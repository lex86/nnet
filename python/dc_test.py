#!/usr/bin/pyton
from dc import *

def start_master(cfg, data, labels):
    master = Master(cfg)
    master.run(data, labels)

def start_worker(cfg):
    worker = Worker(cfg)
    worker.run()

if __name__ == '__main__':
    pass

    
    
