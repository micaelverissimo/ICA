"""
    Implementation of pdf Mixer
"""

import numpy as np
import joblib

class Stats(object):
    
    def __init__(self,n_stats=100,verbose= False):
        self.n_stats = n_stats
        self.verbose = verbose
        
    def Print(self):
        print 'pdf Mixer'
        print 'Number of Samples:' , self.n_stats
        if self.verbose:
            print 'Verbose: True'
        else:
            print 'Verbose: False'