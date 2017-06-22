
# coding: utf-8

# In[1]:

import numpy as np
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
from __future__ import division


class SIR_model:
    """An SIR model to simulate the spread of alpha-syn"""
    
    # constructor
    def __init__(self, v , N_regions, sconn_len, sconn_den, snca, gba, dat, roi_size):
        
        # number of regions
        self.N_regions = N_regions
        
        # store number of normal and misfolded proteins in regions
        self.nor = np.zeros((N_regions, ))
        self.mis = np.zeros((N_regions, ))
        self.nor_history = np.empty((0, N_regions))
        self.mis_history = np.empty((0, N_regions))
        
        # store number of normal and misfoded proteins in paths
        self.path_nor = np.zeros((N_regions, N_regions))
        self.path_mis = np.zeros((N_regions, N_regions))
        self.path_nor_history = np.empty((0, N_regions, N_regions))
        self.path_mis_history = np.empty((0, N_regions, N_regions))
        
        # index of connected components
        (self.idx_x, self.idx_y) = np.nonzero(sconn_len)
        self.non_zero_lengths = sconn_len[np.nonzero(sconn_len)]
        
        # probability of exit a path is set to v/sconn_len
        sconn_len2 = sconn_len.copy()
        sconn_len2[sconn_len2 == 0] = np.inf
        self.prob_exit = v / sconn_len2
        self.prob_exit[sconn_len2==0] = 0
        
        # synthesis rate and  clearance rate
        self.synthesis_rate = norm.cdf(snca) * 0.2
        self.clearance_rate = 1 - np.exp( -norm.cdf(gba))
        #self.stay_rate = norm.cdf(dat)  # yet to be modified .. what the hell is it?!
        
        # get travel weights  --- to be modified 
        self.weights = np.diag(np.sum(sconn_den, axis = 1)) + sconn_den
        self.weights = self.weights / np.sum(self.weights, axis = 1)[np.newaxis].T
        
        # region size
        self.roi_size = roi_size.flatten() / 8
        
        
    def nor_step_region(self):
        """normal alpha-syn growing"""
        self.nor += (self.roi_size * self.synthesis_rate - self.nor * self.clearance_rate)
        
    def nor_travel(self):
        
        # enter paths
        enter_process = self.nor[np.newaxis].T * self.weights
        
        # exit paths
        exit_process = self.path_nor * self.prob_exit
        
        # update paths and regions  ------- feels a bit weird.... to be updated
        self.nor = enter_process[np.nonzero(np.eye(self.N_regions))] + np.sum(exit_process, axis = 0)
        self.path_nor += (enter_process - exit_process)
        self.path_nor[np.eye(self.N_regions)==1] = 0
        
        
    def inject_mis(self, seed, initial_number):
        """inject misfolded alpha-syn in seed region"""
        
        self.mis[seed] = initial_number
        print('now we inject %d misfolded alpha-syn into region %d' % (initial_number, seed))
        
    def mis_travel(self):
        
        # enter paths
        enter_process = self.mis[np.newaxis].T * self.weights
        
        # exit paths
        exit_process = self.path_mis * self.prob_exit
        
        # update paths and regions  ------- feels a bit strange.... to be updated
        self.mis = enter_process[np.nonzero(np.eye(self.N_regions))] + np.sum(exit_process, axis = 0)
        self.path_mis += (enter_process - exit_process)
        self.path_mis[np.eye(self.N_regions)==1] = 0
        
        
    def transmission_region(self, trans_rate):
        
        '"""the transmission process inside regions"""'
        prob_get_infected = 1 - np.exp(-self.mis * trans_rate / self.roi_size)
        # clear process
        self.nor -= self.nor * self.clearance_rate
        self.mis -= self.mis * self.clearance_rate
        
        infected = self.nor * prob_get_infected
        self.nor += (self.roi_size * self.synthesis_rate - infected)
        self.mis += (infected)
        print(self.mis)
        
    def transmission_path(self):
        
        """the transmission process in paths"""
        ### what's the rule of transmission? 
        pass
    
    def record_to_history(self):
        """record the results"""
        self.nor_history = np.append(self.nor_history, self.nor[np.newaxis], axis = 0)
        self.mis_history = np.append(self.mis_history, self.mis[np.newaxis], axis = 0)
        
        self.path_nor_history = np.append(self.path_nor_history, 
                                          self.path_nor.reshape(1, self.N_regions, self.N_regions), axis = 0)
        self.path_mis_history = np.append(self.path_mis_history, 
                                          self.path_mis.reshape(1, self.N_regions, self.N_regions), axis = 0)
        


# In[ ]:



