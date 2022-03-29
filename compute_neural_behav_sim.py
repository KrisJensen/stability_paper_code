#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:19:31 2021
test
@author: kris
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, ttest_1samp, spearmanr
import pickle
from utils import load_rat
from neural_behav_glm import gen_synthetic_rat, sample_data
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

from compute_behav_sim import compute_behav_sim
from compute_neural_sim import compute_neural_sim


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
            #all_corrs.append(spearmanr(s_n, s_b)[0])
            
    return all_sims_n, all_sims_b, all_corrs
