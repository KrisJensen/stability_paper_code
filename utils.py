#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:09:46 2021

@author: kris
"""

import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
import pickle

global_params = {'smooth_peth': True, #convolve peth w/ gaussian
                'minspike': 10, # minimum number of spikes in a session
                'min_sameday': 0.15, #minimum same-day similarity
                'constrain_intercept': True, #constrain intercept to be in [-1, 1]
                'min_dt': 3, #minimum time difference to compute alpha (recording time is this +1)
                'min_datapoints': 4, #minimum number of datapoints to fit alpha
                'conv_2tap': 15./1000., #Gaussian convolution width for twotap (in s)
                'conv_wds': 10./1000., #Gaussian convolution width for wds (in s)
                'binsize': 20./1000., #binsize (in s)
                }

def check_modulated(us, same_sims, name):
    '''check that neurons are significantly modulated'''

    if global_params['min_sameday'] < -1: #allow all units
        return np.arange(len(us)), np.array(us)

    unums = []
    inds = []
    for iu, unum in enumerate(us):
        if unum in same_sims[name].keys():
            if same_sims[name][unum] >= global_params['min_sameday']:
                inds.append(iu)
                unums.append(unum)

    return np.array(inds), np.array(unums)

def fit_exp_f(params, X, y, L1 = False, baseline = False):
    if global_params['constrain_intercept']:
        res = y - np.tanh(params[0])*np.exp(params[1]*X) #in [-1, 1]
    else:
        res = y - params[0]*np.exp(params[1]*X)
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

def calc_exp_ind(dts, sims, return_res = False):
    """
    Fits an exponential model to similarity as a function of time difference
    Return stability index k = -1/tau s.t. sim(dt) = a*exp(k*t)
    dts: time differences
    sims: similarities
    thresh_i: minimum baseline (day 0) similarity 'a'
    """
    res = fit_model(dts, sims)
    
    i, s = res.x
    if res.success:
        return s, res if return_res else s
    else: #ignore a very small fraction of resamples where the model does not converge
        return np.nan, res if return_res else np.nan
    
def combine_rats(rats, name = 'combined', offset = 10000):
    rat = rats[0]
    rat['name'] = name
    for irat, newrat in enumerate(rats[1:]):
        shift = int((irat+1)*offset)
        for unum, u in newrat['units'].items(): #add ephys info
            rat['unittypes'][unum + shift] = newrat['unittypes'][unum] #unit type
            rat['units'][unum + shift] = {daynum + shift: day for (daynum, day) in u.items()} #unit data
            
        for daynum, day in newrat['trials'].items(): #add behavioral info
            rat['trials'][daynum + shift] = day
        
    return rat
    
def load_rat(ratname, parsed = True, trim = False):
    
    ext = '_warped' if parsed else ''
    if trim:
        ext = '_trimmed'
        print("loading trimmed data")
        
    if ratname == 'DLS':
        names = ['Hindol', 'Jaunpuri', 'Dhanashri']
    elif ratname == 'MC':
        names = ['Hamir', 'Gandhar', 'Gorakh']
    elif ratname == 'DLS_wds':
        names = ['Hindol_wds', 'Jaunpuri_wds', 'Dhanashri_wds']
    elif ratname == 'MC_wds':
        names = ['Hamir_wds', 'Gandhar_wds', 'Gorakh_wds']
    else:
        names = [ratname]
    
    rats = []
    for name in names:
        newrat = pickle.load(open('./data/'+name+'_data'+ext+'.p', 'rb'))
        rats.append(newrat)       
            
    if len(rats) == 1:
        return rats[0]
    
    else: #combine data across rats
        return combine_rats(rats, name = ratname)
    
def asymptotic_model(params, X):
    return -np.abs(params[0]) - np.abs(params[1])*np.exp(-np.abs(params[2])*X)

def asymptotic_residual(params, X, y, L1 = False):
    yhat = asymptotic_model(params, X)
    res = np.abs(y-yhat)
    return np.mean(res)

def asymptotic_mean_residual(params, X, y, L1 = False):
    yhat = asymptotic_model(params, X)
    res = (y-yhat)**2
    return np.mean(res)

def fit_asymptotic_model(ts, alphas):
    """
    Fits an exponential model y = a + b*exp(c*t) ~ a + b + b*c*t
    """
    ts, alphas = np.array(ts), np.array(alphas)
    m = np.mean(alphas)
    s, i = linregress(ts, alphas-m)[:2] #initialize frroom linear
    
    ### first fit from linear initialization ###
    a, b, c = m, i, s
    res_lin = minimize(asymptotic_residual, [a, b, c], args = (ts, alphas), options = {'maxiter': 1e3}, method = 'L-BFGS-B')
    
    ### now fit from zero initialization ###
    a, b, c = 0., 0., 0.
    res_zero = minimize(asymptotic_residual, [a, b, c], args = (ts, alphas), options = {'maxiter': 1e3}, method = 'L-BFGS-B')
    
    ### now fit from mean initialization ###
    res_mean = minimize(asymptotic_mean_residual, [a, b, c], args = (ts, alphas), options = {'maxiter': 1e3}, method = 'L-BFGS-B')
    res_mean = minimize(asymptotic_residual, res_mean.x, args = (ts, alphas), options = {'maxiter': 1e3}, method = 'L-BFGS-B')
    
    if not any([res_lin.success, res_zero.success, res_mean.success]):
        return [np.nan]*3 #return nan in the unlikely event that no initialization converges
    
    ress = [res_lin, res_zero, res_mean]
    errs = [asymptotic_residual(res.x, ts, alphas) if res.success else np.inf for res in ress]
    
    return np.abs(ress[np.argmin(errs)].x)


np.seterr(all="ignore")
    
def get_modes(ratname):
    
    if 'wds' in ratname: return [1]
    if ratname == 'JaunpuriL': ratname = 'Jaunpuri'
    a = {
            'Hindol':[4],
            'Dhanashri':[2],
            'Jaunpuri':[2],
            'Hamir':[2],
            'Gandhar':[1],
            'Gorakh':[1]
            }
    
    return a[ratname]
    
    