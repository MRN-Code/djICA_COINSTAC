#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 00:07:13 2018

@author: Jafar M 
"""

import numpy as np
from scipy.stats import ortho_group
from scipy import signal


def generate_synthetic(typ): 

    n_ind_compo = 3             # number of independent components
    # for method 2 this is fix to 3
    n_samples = 2000            # number of samples
    D = 6              # dimension: numer of sensors 
    
    
    num_sites = 2
    num_samples_site = np.int(n_samples / num_sites)
    st_id = 0
    en_id = num_samples_site
    
    
    if typ == 1: 
        A = ortho_group.rvs(dim = D)
        A = A[:, :n_ind_compo]        # mixing matrix
        
        # Data generation
        mn = np.zeros([D,])
        cv = np.eye(D)
        h = np.random.exponential(scale = 2, size = [n_ind_compo, n_samples])
        X = np.dot(A, h) + np.sqrt(0.0005) * np.random.multivariate_normal(mn, cv, n_samples).T
    
    elif typ == 2: 
        
        time = np.linspace(0, 8, n_samples)
        
        
        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
        
         
        
        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise
        
        S /= S.std(axis=0) # Standardize data
        # Mix data
        #A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
        A= ortho_group.rvs(dim = D)
        A = A[:, :3]  
        X = np.dot(A, S.T)    
#    
    

    
    C = (1.0 /n_samples) * np.dot(X, X.T)
    U, S, V = np.linalg.svd(C)
    Uk = U[:, :n_ind_compo]
    
    
    for s in range(num_sites):
        Xs = np.array(X[:, st_id : en_id])
        st_id += num_samples_site
        en_id += num_samples_site
        filename = 'value' + str(s) + '.npz'
        np.savez(filename, X= Xs, U=Uk, K = int(n_ind_compo) )

    return X, A, n_ind_compo
#%%
X, A, n_ind_compo = generate_synthetic(2)