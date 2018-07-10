#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 23:02:04 2018

@author: aaco
"""

import json
import numpy as np
import sys
from ancillary import list_recursive


#%%
def remote_ica(args):
    
    input_list = args["input"]
    
    
    #n_site = len(input_list)
    
    # initial values
    blowUp_thr = 1e8
    maxItr = 10
    no_change = 1e-8
    
    # recover the iteration values remote cash
    
    U = np.array(input_list['local0']['U'])
    itr = args['input']['local0']['iter']
    W = np.array(args['input']['local0']['W'])
    b = np.array(args['input']['local0']['b'])
    rho = args['input']['local0']['rho']
    
    K = len(W)   # num indpendent components num_ind_compo
    
    # aggregation 
    
    gradSum = 0.0
    biasGradSum = 0.0    
    
    gradSum = np.sum(np.array(input_list[site]['G']) for site in input_list)
    biasGradSum = np.sum(np.array(input_list[site]['h']) for site in input_list)
    
#    for i in range(0, n_site):
#        gradSum += np.array(args[i]['G'])
#        biasGradSum += np.array(args[i]['h'])
    
    # stopping rule 
    
    if itr < maxItr and np.linalg.norm(gradSum) > no_change:
    #    sys.stderr.write("\n Updating W")    
        W = np.add(W, gradSum, casting='unsafe')
    #    sys.stderr.write("\n Updating b")    
        b = np.add(b, biasGradSum, casting='unsafe')
        itr += 1
    
    #    check blowout and update rho if needed
        if np.max(np.abs(W)) >= blowUp_thr:
    #      sys.stderr.write("\n Blowout detected. Restarting...")    
            rho = rho * 0.8
    #      initialize W and b again
            #W = np.eye(K)
            #W = np.random.uniform(-1,1,(K,K))
            W = np.random.normal(0,0.5,(K,K))
            #b = np.zeros([K, 1])
            b = np.random.normal(0,1,(K,1))
            itr = 1
       
        computation_output = {'output': 
                                   {'W' : W.tolist(), 'b' : b.tolist(), 
                                         'rho' : rho, 'iter' : itr },
                                   
                              'cache': {'W' : W.tolist(), 'b' : b.tolist(), 
                                         'rho' : rho, 'iter' : itr, 'U': U.tolist()}    
                                   }
                                   
    else:
        itr += 1
        A_est = np.linalg.pinv(np.dot(W, U.T))

    
        computation_output = {'output': 
                                {'W' : W.tolist(),  
                                    'iter' : itr,'mixingMat': A_est.tolist(),
                                           }, 'success': True }
                                   
        
        
    return json.dumps(computation_output)

#%%
if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = remote_ica(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")



