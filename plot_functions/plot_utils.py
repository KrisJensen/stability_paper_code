#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:39:56 2021

@author: kris
"""


import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize

col_dls = [0.466, 0.674, 0.188]
col_mc = [0.301, 0.745, 0.933]
col_stab = [0.85, 0.325, 0.098]
col_un = [0.929, 0.694, 0.125]


col_dls = [0, 0.447, 0.741]
col_mc = [0.85, 0.325, 0.098]

col_stab = [0.466, 0.674, 0.188]
col_un = [0.494, 0.184, 0.556]

col_blue = np.array([56, 84, 160]) / 256
col_green = np.array([66, 140, 84]) / 256

global_cmap = 'coolwarm'

def get_col(name):
    cols = {'DLS': col_dls,
            'Hindol': col_dls,
            'Dhanashri': col_dls,
            'Jaunpuri': col_dls,
            'MC': col_mc,
            'Hamir': col_mc,
            'Gandhar': col_mc,
            'Gorakh': col_mc}
    
    keys = list(cols.keys())
    for k in keys:
        cols[k+'_wds'] = cols[k]
    
    return cols[name]

global_params = {'constrain_intercept': True}

panel_font = 14
png_dpi = 1200

def fit_exp_f(params, X, y, L1 = False, baseline = False):
    if global_params['constrain_intercept']:
        res = y - np.tanh(params[0])*np.exp(params[1]*X) #in [-1, 1]
    else:
        res = y - params[0]*np.exp(params[1]*X)
    #return np.mean(np.abs(res))
    if baseline:
        res = res - params[2] #subtract baseline

    return np.mean(res**2)


def fit_model(dts, sims, baseline = False):
    dts, sims = np.array(dts), np.array(sims)
    s, i = linregress(dts, sims)[:2]
    
    if global_params['constrain_intercept']:
        i = max(min(i, 0.95), -0.95) #in [-0.95, 0.95]
        i = np.arctanh(i) #transfer function
    
    if baseline: #include baseline
        def new_f(params, X, y): return fit_exp_f(params, X, y, baseline = True)
        res = minimize(new_f, [i-np.mean(sims), s, np.mean(sims)], args = (dts, sims), options = {'maxiter': 1e3}, method = 'L-BFGS-B')
    else:    
        res = minimize(fit_exp_f, [i, s], args = (dts, sims), options = {'maxiter': 1e3}, method = 'L-BFGS-B')
    
    if global_params['constrain_intercept']:
        res.x[0] = np.tanh(res.x[0])
    
    return res


np.seterr(all="ignore")
