#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:04:40 2021

@author: kris
"""
import numpy as np
from scipy.stats import pearsonr, binned_statistic, sem, poisson
from utils import fit_asymptotic_model, calc_exp_ind, global_params

def bootstrap_asymptotic(ts, alphas, N = 1000, prop = True):
    ts, alphas = np.array(ts), np.array(alphas)
    ndat = len(ts)
    result = np.zeros((N, 3))
    ps = ts/np.sum(ts)
    for n in range(N):
        if prop:
            inds = np.random.choice(np.arange(ndat), ndat, replace = True, p = ps)
        else:
            inds = np.random.choice(ndat, ndat, replace = True)
        res = fit_asymptotic_model(ts[inds], alphas[inds])
        result[n, :] = res
    return result

def get_peths(u, days, synthetic = False):
    if synthetic: #use GLM generated data
        if all([u[day]['glm'] for day in days]):
            hs = [u[day]['glm_peth_w'] for day in days]
        else:
            hs = [-np.inf*np.ones(45) for _ in days]
    else: #use experimental data
        hs = [u[day]['peth_w'] for day in days]
    return hs

def compute_neural_sim(rat, synthetic = False, day0 = True):
    unums = []
    day1s = []
    dts = []
    sims = []
    
    for unum, u in rat['units'].items():
        newdt, newsim, newday1 = [], [], []
        days = np.sort(list(u.keys()))
        if not day0:
            days = days[1:]
        
        for i1, day1 in enumerate(days):
            for day2 in days[i1+1:]:

                h1, h2 = get_peths(u, [day1, day2], synthetic)
                count = True if (min(sum(h1), sum(h2)) >= global_params['minspike']) else False
                if count: sim = pearsonr(h1, h2)[0]
                    
                if count:
                    newsim.append(sim)
                    newdt.append(day2-day1)
                    newday1.append(day1)
                        
        if len(newdt) >= 1:
            dts.append(newdt)
            sims.append(newsim)
            day1s.append(newday1)
            unums.append(unum)
                    
    return unums, dts, sims, day1s


def compute_shuffled_sims(rat, unums, nsamples = 10000):
    '''compute a lower bound for a given set of units by sampling PETHs on non identical days'''
    sims = np.zeros(nsamples)
    n = 0
    unums = np.array(unums)
    while n < nsamples:
        unum1 = np.random.choice(unums)
        diffs = np.abs(unums-unum1)
        same_us = unums[ (diffs > 0.5) & (diffs < 5000) ] #from same animal

        if len(same_us) > 0:
            unum2 = np.random.choice( same_us )
            u1, u2 = [rat['units'][unum] for unum in [unum1, unum2]]
            day1, day2 = [np.random.choice(list(u.keys())) for u in [u1, u2]]
            h1 = u1[day1]['peth_w']
            h2 = u2[day2]['peth_w']

            count = True if (min(sum(h1), sum(h2)) >= global_params['minspike']) else False
            if count:
                sim = pearsonr(h1, h2)[0]
                sims[n] = sim
                n += 1
    return sims

def calc_alphas(unums, dts, sims, comp_error = False):
    alphas = []
    inds = []
    ress = []
    errors = []
    mean_errors = []
    all_ress = []
    for iu, unum in enumerate(unums):
        dt, sim = dts[iu], sims[iu]
        if len(dt) >= global_params['min_datapoints'] and np.amax(dt) >= global_params['min_dt']:
            s, res = calc_exp_ind(dt, sim, return_res = True)
            all_ress.append(res)
            if not np.isnan(s):
                alphas.append(s)
                inds.append(iu)
                
                if comp_error:
                    ### also compute model error ###
                    ress.append(res)
                    residual = np.array(sim) - res.x[0]*np.exp(res.x[1]*np.array(dt))
                    errors.append(np.mean(np.abs(residual)))
                    
                    mean_residual = np.array(sim) - np.mean(np.array(sim))
                    mean_errors.append(np.mean(np.abs(mean_residual)))
                    
    if comp_error:
        return np.array(alphas), np.array(inds), ress, errors, mean_errors, all_ress
    else:
        return np.array(alphas), np.array(inds)
    

def binning(xs, ys, bins = 8, bins1 = None, stat = 'median'):
    xmin, xmax = np.amin(xs), np.amax(xs)
    if bins1 is None:
        bins1 = np.linspace(xmin, xmax, bins)
    else:
        bins1 = np.array(bins1)
    bins2 = 0.5*(bins1[1:] + bins1[:-1])
    allbins = [bins1, bins2]
    allbins = [bins1]+[frac*bins1[1:]+(1-frac)*bins1[:-1] for frac in [0.5]]
    y = [binned_statistic(xs, ys, statistic = stat, bins = bins)[0] for bins in allbins]
    s = [binned_statistic(xs, ys, statistic = sem, bins = bins)[0] for bins in allbins]
    x = [0.5*(bins[1:] + bins[:-1]) for bins in allbins]
    x = [binned_statistic(xs, xs, statistic = 'mean', bins = bins)[0] for bins in allbins]
    x, y, s = [np.concatenate(dat) for dat in [x, y, s]]
    args = np.argsort(x)
    x, y, s = x[args], y[args], s[args]
    return x, y, s
