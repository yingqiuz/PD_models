
# coding: utf-8

# In[1]:

import numpy as np
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
from __future__ import division


# In[2]:

class agent_based_model:
    
    """set the model as object"""
    
    # constructor
    def __init__(self, v , N_regions, sconn_len, sconn_den, snca, gba, dat, roi_size):
        
        # number of regions
        self.N_regions = N_regions
        
        # store number of normal and misfolded proteins in regions
        self.nor = np.zeros((N_regions,), dtype = np.int)
        self.nor_history = np.empty((0, N_regions), dtype = np.int)
        self.mis = np.zeros((N_regions,), dtype = np.int)
        self.mis_history = np.empty((0, N_regions), dtype = np.int)
        
        
        # store number of normal and misfolded proteins in paths
        self.sconn_len = np.int_(np.ceil(sconn_len / v))
        (self.idx_x, self.idx_y) = np.nonzero(self.sconn_len)
        self.non_zero_lengths = self.sconn_len[np.nonzero(self.sconn_len)]  # int64
        self.path_nor = [ [[] for y in range(N_regions)] for x in range(N_regions)]
        
        #### is there more efficient way to do this?  --- to be updated.......
        for x, y, v in zip(self.idx_x, self.idx_y, self.non_zero_lengths):
            self.path_nor[x][y] = [0 for k in range(v)]
            
        ## create self.path_mis    
        self.path_mis = [ [[] for y in range(N_regions)] for x in range(N_regions)]
        
        #### is there more efficient way to do this?  --- to be updated.......
        for x, y, v in zip(self.idx_x, self.idx_y, self.non_zero_lengths):
            self.path_mis[x][y] = [0 for k in range(v)]
            
        # record the trajectory
        self.path_nor_history = []
        self.path_mis_history = []
        
        # continuous path and path history
        self.path_nor_cont = np.zeros((N_regions, N_regions), dtype = np.int)
        self.path_mis_cont = np.zeros((N_regions, N_regions), dtype = np.int)
        self.path_nor_cont_history, self.path_mis_cont_history = [np.empty((0, self.N_regions, self.N_regions))] * 2
        
        
        # synthesis rate and clearance rate
        self.synthesis_rate = norm.cdf(snca) * 0.5
        self.clearance_rate = 1 - np.exp(-norm.cdf(gba))
        self.stay_rate = 1-norm.cdf(dat)  # yet to be modified .. what the hell is it?!
        
        
        # probability of exit a path is set to v/sconn_len
        sconn_len2 = sconn_len.copy()
        sconn_len2[sconn_len2 == 0] = np.inf
        self.prob_exit = v / sconn_len2
        self.prob_exit[sconn_len2==0] = 0
        
        # get travel weights  --- to be modified 
        # with DAT1
        #self.weights = np.diag(self.stay_rate * np.sum(sconn_den, axis = 1)) + sconn_den
        
        # without DAT1
        self.weights = np.max(np.sum(sconn_den, axis = 1))* 0.2 * np.eye(self.N_regions) + sconn_den
        
        # scale 
        self.weights = self.weights / np.sum(self.weights, axis = 1)[np.newaxis].T
        
        # region size
        self.roi_size = np.int_(np.round(roi_size.flatten()/8))
        
    def nor_step_region(self):
        """step: normal alpha-syn being synthesized and cleared in regions"""
        ### synthesis and clearance first
        ## clearance
        self.nor -= np.array([np.sum(np.random.uniform(0, 1, (v, )) < k) 
                             for k, v in zip(self.clearance_rate, self.nor)])
        
        
        ## synthesis
        self.nor += np.array([np.sum(np.random.uniform(0, 1, (v, )) < k) 
                              for k, v in zip(self.synthesis_rate, self.roi_size)])
    
    def nor_travel(self):
        """step: normal alpha-syn traveling in paths"""
        # alpha syn  -- from region to path
        # exit region
        exit_process = np.array([ np.random.multinomial(self.nor[k], self.weights[k]) 
                                 for k in range(self.N_regions)], dtype = np.int)
        
        # alpha syn -- from path to region
        # enter region
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)
        for x, y in zip(self.idx_x, self.idx_y):
            # fetch then remove the last element
            enter_process[x, y] = self.path_nor[x][y].pop()
            # update paths
            self.path_nor[x][y].insert(0, exit_process[x, y])

        # update regions 
        self.nor = exit_process[np.nonzero(np.eye(self.N_regions))] + np.sum(enter_process, axis = 0)
        
    def nor_travel_cont(self):
        
        # exit regions:
        exit_process = np.array([ np.random.multinomial(self.nor[k], self.weights[k]) 
                                 for k in range(self.N_regions)], dtype = np.int)
        
        # enter regions:
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype = np.int)
        
        for x, y in zip(self.idx_x, self.idx_y):
            enter_process[x, y] = np.sum(np.random.uniform(0, 1, (self.path_nor_cont[x, y], ) 
                                                          ) )< self.prob_exit[x, y]
        
        # update:
        self.path_nor_cont += (exit_process - enter_process)
        self.path_nor_cont[np.eye(self.N_regions)==1] == 0
        self.path_nor = exit_process[np.nonzero(np.eye(self.N_regions))] + np.sum(enter_process, axis = 0)
        
    
        
    def inject_mis(self, seed, initial_number):
        
        """inject initual_number misfolded protein into seed region"""
        # initial_number must be an interger
        self.mis[seed] = initial_number
        print('Now we injected %d misfolded alpha-syn into region %d' % (initial_number, seed))
    
    
    def mis_travel_cont(self):
        # exit regions:
        exit_process = np.array([ np.random.multinomial(v, self.weights[k]) 
                                 for k, v in enumerate(self.mis)], dtype = np.int)
        
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype = np.int)
        
        # enter regions:
        for x, y in zip(self.idx_x, self.idx_y):
            enter_process[x, y] = np.sum(np.random.uniform(0, 1, (self.path_mis_cont[x, y], )
                                                          ))< self.prob_exit[x, y]
        
        # update
        self.path_mis_cont += (exit_process - enter_process)
        self.path_mis_cont[np.eye(self.N_regions)==1] == 0
        self.mis = np.sum(enter_process, axis = 0) + exit_process[np.nonzero(np.eye(self.N_regions))]
    
    def mis_travel(self):
        """ step in paths for normal and misfolded alpha syn"""
        # exit regions
        exit_process = np.array([ np.random.multinomial(v, self.weights[k]) 
                                 for k, v in enumerate(self.mis)], dtype = np.int)
        # alpha syn -- from path to region
        # enter region
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype = np.int)
        for x, y in zip(self.idx_x, self.idx_y):
            # fetch then remove the last element
            enter_process[x, y] = self.path_mis[x][y].pop()
            # update paths
            self.path_mis[x][y].insert(0, exit_process[x, y])

        # update regions 
        self.mis = exit_process[np.nonzero(np.eye(self.N_regions))] + np.sum(enter_process, axis = 0)
        
    def transmission_region(self):
        """clearance and synthesis of normal/misfolded alpha-syn/ transsmssion process in regions"""
        ## clearance
        cleared_nor = np.array([np.sum(np.random.uniform(0, 1, (v, )) < k) 
                             for k, v in zip(self.clearance_rate, self.nor)])
        
        cleared_mis = np.array([np.sum(np.random.uniform(0, 1, (v, )) < k) 
                             for k, v in zip(self.clearance_rate, self.mis)])
        
        self.prob_infected = 1 - np.exp(- (self.mis / (self.roi_size)) )
        # the remaining after clearance
        self.nor -= cleared_nor
        self.mis -= cleared_mis
        #self.prob_infected = 1 - np.exp(- (self.mis / self.roi_size) )
        infected_nor = np.array([np.sum(np.random.uniform(0, 1, (v, )) < self.prob_infected[k]) 
                             for k, v in enumerate(self.nor)])
        # update self.nor and self.mis
        self.nor += (np.array([np.sum(np.random.uniform(0, 1, (v, )) < self.synthesis_rate[k]) 
                              for k, v in enumerate(self.roi_size)]) - infected_nor)
        self.mis += infected_nor 
        
        print(self.mis)
        
    def tranmission_path(self, trans_rate_path):
        
        """transmission process in paths"""
        for x, y, v in zip(self.idx_x, self.idx_y, self.non_zero_lengths):
            ### perhaps trans_rate_path should be set to 1/v ?
            # transmission rate is scaled by exp(distance) in voxel space
            rate_get_infected = (self.path_mis[x][y] * trans_rate_path) / (self.roi_size * 
                                                       np.exp(np.absolute(np.arange(v) - 
                                                                          np.arange(v)[np.newaxis].T)) )
            prob_get_infected = 1 - np.exp(np.sum(-rate_get_infected, axis = 1))
            infected_path = np.array([np.sum(np.random.uniform(0, 1, (k, ))<v) for 
                                      k, v in zip(self.path_nor[x][y], prob_get_infected)])
            
            # update self.path_nor and self.path_mis
            self.path_nor[x][y] -= infected_path
            self.path_mis[x][y] += infected_path
            
        
    def record_to_history(self):
        """record the results of each step into the recorder"""
        self.nor_history = np.append(self.nor_history, self.nor[np.newaxis], axis = 0)
        self.mis_history = np.append(self.mis_history, self.mis[np.newaxis], axis = 0)
        
        self.path_nor_history.append(self.path_nor)
        self.path_mis_history.append(self.path_mis)
        


# In[ ]:



