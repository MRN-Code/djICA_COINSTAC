#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 22:21:43 2018

@author: JAFAR Mohammadi (Aco)
"""

import json
import os
import sys
import numpy as np
from ancillary import list_recursive

#%% Sigmoid function 
def mySigmoid(X):
    tmp = 1 + np.exp(-X)
    return np.divide(1, tmp)



#%% local function 
def local_ica(args): 
    
    # initialization and reading the files X and Uk, first step
    if not 'iter' in args['input']: 
        
        input_list = args["input"]
        myFile = input_list["samples"]
    
        # read the local data X and subspcaes U 
        filename = os.path.join(args["state"]["baseDirectory"], myFile)
        tmp = np.load(filename)
        # data
        X = tmp['X']           
        # K Princeple subspaces 
        U = tmp['U']
        # K number of priceple subspaces
        K = tmp['K']
        
        # initialization 
        D, N = X.shape
        block = 1 # (N/20) ** (0.5)
        ones = np.ones([N, 1])
        
        W = np.eye(K)
        b = np.zeros([K, 1])
        rho = 0.0115 / np.log(D)
        itr = 1      
        
        # project data on the reduced subspace
       
        Xred = np.dot(U.T, X)
        BI = block * np.eye(Xred.shape[0])
        
        # take grdient step
        Z = np.dot(W, Xred) + np.dot(b, ones.T)
        Y = mySigmoid(Z)
        G = rho * np.dot(BI + np.dot(1 - 2*Y, Z.T), W)
        h = rho * np.sum(1 - 2*Y, axis = 1)
        h = h.reshape([Xred.shape[0], 1])
    
        rho = rho/(2*itr)
        
        
        computation_output = {
            "output": {
                'G' : G.tolist(), 'h' : h.tolist(),
                         'rho' : rho, 'W' : W.tolist(), 'b' : b.tolist(),
                         'iter': itr, 'U': U.tolist()
                      }, 
            
            "cache": { 'Xred': Xred.tolist(), 'U': U.tolist(), 'N': N, 'D': D}
                    
                }
        
    else: 
        # iteration > 1 this will be run
        
        input_list = args['input']
        Xred = np.array(args['cache']['Xred'])
        U = np.array(args['cache']['U'])
        N = args['cache']['N']
        D= args['cache']['D']
        
        block = 1# (N/20) ** (0.5)
        ones = np.ones([N, 1])
        BI = block * np.eye(Xred.shape[0])
        
        # extract the data from the remote
        W = np.array(input_list['W'])
        b = np.array(input_list['b'])
        rho = input_list['rho']
        itr = input_list['iter']
        
        # take grdient step
        Z = np.dot(W, Xred) + np.dot(b, ones.T)
        Y = mySigmoid(Z)
        G = rho * np.dot(BI + np.dot(1 - 2*Y, Z.T), W)
        h = rho * np.sum(1 - 2*Y, axis = 1)
        h = h.reshape([Xred.shape[0], 1])
    
        rho = rho/(2*itr)
        
        computation_output = {
            "output": {
                'G' : G.tolist(), 'h' : h.tolist(),
                         'rho' : rho, 'W' : W.tolist(), 'b' : b.tolist(),
                         'iter': itr, 'U': U.tolist()
                      }, 
            
            "cache": { 'Xred': Xred.tolist(), 'U': U.tolist(), 'N': N, 'D': D }
                     
                }
        
        
    return json.dumps(computation_output)

#%%
if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_ica(parsed_args)
        sys.stdout.write(computation_output)

    else:
        raise ValueError("Error occurred at Local")

   
        
        
        
        
        
