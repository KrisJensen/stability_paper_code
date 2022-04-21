#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:39:29 2021

@author: kris
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import pickle

def compute_behav_sim(rat, verbose = True):

    #note that we only use data from 0.1s before the first tap to 0.1s after the second tap
    #'inds' are the timebins corresponding to this behavior
    inds = np.arange(48, 157)

    if 'wds' in rat['name']:
        limbs = ['acc']
        ncoords = 3
    else:
        limbs = ['paw_L', 'paw_R']
        ncoords = 2

    day1s = []
    dts = []
    sims = []
    days = np.sort(list(rat['trials'].keys()))
    for i1, day1 in enumerate(days):
        for day2 in days[i1+1:]:
            newsims = []
            for limb in limbs:
                for coord in range(ncoords):
                    if 'wds' in rat['name']:
                        v1, v2 = [np.mean(rat['trials'][day]['kinematics_w'][limb][:, :, coord], axis = 0) for day in [day1, day2]] #consider all time points for this
                    else:
                        v1, v2 = [np.mean(rat['trials'][day]['vels_w'][limb][:, inds, coord], axis = 0) for day in [day1, day2]] #timepoints
                    newsims.append(pearsonr(v1, v2)[0])
            sims.append(np.mean(newsims))
            #time difference here is just the number of days separating to sessions
            dts.append(day2-day1)
            day1s.append(day1)
        if verbose: print(day1, 'of', str(days[-1])+':', np.mean(sims))
        
    return dts, sims, day1s

