#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:19:31 2021
test
@author: kris
"""
import numpy as np
from scipy.stats import pearsonr

def comp_neural_behav_sim(unums, dts_n, sims_n, day1s_n, dts_b, sims_b, day1s_b, permute = False):
    inds_b = (np.array(dts_b) == 1)
    sims_b, day1s_b = np.array(sims_b)[inds_b], np.array(day1s_b)[inds_b]
    b_dict = {day: i for (i, day) in enumerate(day1s_b)}
    if permute: #permute behavioral data
        np.random.shuffle(sims_b)

    all_sims_n, all_sims_b, all_corrs = [], [], []
    for iu, unum in enumerate(unums):
        dt, sim, day1 = [np.array(arr) for arr in [dts_n[iu],  sims_n[iu], day1s_n[iu]]]
        if sum(dt == 1) >= 4:
            inds = (dt == 1)
            d1 = day1[inds]
            s_n = sim[inds]
            s_b = [sims_b[b_dict[d]] for d in d1]    
            all_sims_n.append(s_n)
            all_sims_b.append(s_b)
            all_corrs.append(pearsonr(s_n, s_b)[0])
            
    return all_sims_n, all_sims_b, all_corrs
