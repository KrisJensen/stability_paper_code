#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:04:40 2021

@author: kris
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, binned_statistic, sem, poisson
from scipy.optimize import minimize
import pickle
from utils import load_rat, fit_asymptotic_model, asymptotic_model, calc_exp_ind, fit_model, global_params, check_modulated
from neural_behav_glm import gen_synthetic_rat, sample_data

def bootstrap_asymptotic(ts, alphas, N = 1000, prop = True):
    ts, alphas = np.array(ts), np.array(alphas)
    ndat = len(ts)
    result = np.zeros((N, 3))
    ps = ts/np.sum(ts)
    for n in range(N):
        if n % 500 == 0:
            print(n)
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

def get_rates(u, days, synthetic = False):
    if synthetic: #use GLM generated data
        raise NotImplementedError
    else: #use experimental data
        rs = [np.sum(np.mean(u[day]['peth_w_t'], axis = 0)) for day in days] #rate in spikes per trial
        counts = [np.sum(u[day]['peth_w_t']) for day in days]
        trials = [u[day]['peth_w_t'].shape[0] for day in days]
    return rs, counts, trials

def calc_pois_sim(c1, c2, t1, t2):
    r1, r2 = c1/t1, c2/t2
    imin, imax = np.argmin([r1, r2]), np.argmax([r1, r2]) #min and max rates
    
    rate = (c1+c2)/(t1+t2) #optimal rate is total rate, measured per trial
    
    ### consider cdf up to lambda1
    cdf1 = poisson.cdf([c1, c2][imin]+0.1, [t1, t2][imin]*rate, loc=0)
    cdf2 = 1-poisson.cdf([c1, c2][imax], [t1, t2][imax]*rate, loc=0)
    return cdf1+cdf2

def calc_pois_sim(c1, c2, t1, t2):
    cs, ts = np.array([c1, c2]), np.array([t1, t2])
    rs = cs/ts
    args = np.argsort(ts)
    cs, ts, rs = cs[args], ts[args], rs[args]
    rate = sum(cs)/sum(ts)
    delta = np.abs(rs[1] - rs[0]) #rate difference
    
    ### let c1 \in [0.5*lambda, 1.5*lambda]
    cmin, cmax = int(np.floor(0.5*rate*ts[0]-4)), int(np.ceil(1.5*rate*ts[0]+4))
    tile_cs = np.arange(cmin, cmax)
    
    if cs[1] > 100:
        tile_cs = tile_cs[::5]
    if cs[1] > 300:
        tile_cs = tile_cs[::15]
    elif cs[1] > 1000:
        tile_cs = tile_cs[::50]
    
    ps = np.zeros(len(tile_cs))
    cdfs = np.zeros(len(tile_cs))
    for i, c in enumerate(tile_cs):
        ps[i] = poisson.pmf(c, rate*ts[0]) #p(c1)
        
        cmin = ts[1]*(c/ts[0] - delta)
        cmax = ts[1]*(c/ts[0] + delta)

        cdfs[i] = poisson.cdf(cmin, rate*ts[1]+1e-5) #c2 smaller than or equal to t2*(c1/t1-delta)
        cdfs[i] += (1 - poisson.cdf(cmax, rate*ts[1]+1e-5)) #c2 larger than or equal to t2*(c1/t1-delta)
        
        #print(rate*ts[1], c, cmin, cmax, ps[i], cdfs[i], (1 - poisson.cdf(cmax, rate*ts[1]+1e-5)))
        
    ps /= np.sum(ps) #normalize
    return np.sum(ps*cdfs)

def compute_neural_sim(rat, synthetic = False, day0 = True, rate = False):
    
    unums = []
    day1s = []
    dts = []
    sims = []
    
    for unum, u in rat['units'].items():
        newdt, newsim, newday1 = [], [], []
        days = np.sort(list(u.keys()))
        #print(unum)
        if not day0:
            days = days[1:]
        
        for i1, day1 in enumerate(days):
            for day2 in days[i1+1:]:
                
                if rate:
                    [r1, r2], [c1, c2], [t1, t2] = get_rates(u, [day1, day2], synthetic)
                    count = True
                    if max(r1, r2) > 0:
                        sim = 1 - 1*np.abs(r1-r2)/(r1+r2)
                    else: #two silent sessions are similar
                        sim = 1
                    ### joint probability under the optimal poisson model ###
                    sim = calc_pois_sim(c1, c2, t1, t2)
                    #if min(c1, c2) < global_params['minspike']: count = False
                        
                else:
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

        #unum1, unum2 = np.random.permutation(unums)[:2] #select two random units

        unum1 = np.random.choice(unums)
        diffs = np.abs(unums-unum1)
        same_us = unums[ (diffs > 0.5) & (diffs < 5000) ] #from same animal

        if len(same_us) > 0:
            unum2 = np.random.choice( same_us )
            #print(unum1, unum2)

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
    # print(len(x), len(y), len(s))
    print([np.histogram(xs, bins)[0] for bins in allbins])
    args = np.argsort(x)
    x, y, s = x[args], y[args], s[args]
    
    return x, y, s

def rolling_median(xs, ys, wsize = None, delta = 1, xmax = None):
    if wsize is None:
        wsize = (np.amax(xs) - np.amin(xs))/10
    if xmax is None:
        xmax = np.amax(xs)
    
    xs, ys = np.array(xs), np.array(ys)
    
    x0s = np.arange(np.amin(xs), xmax-wsize, delta)
    x1s = x0s+wsize
    x0s = np.concatenate([x0s, np.ones(1)*(xmax-wsize)])
    x1s = np.concatenate([x1s, np.ones(1)*(np.amax(xs))])
    
    # print(np.amin(x0s), np.amax(x1s), np.amin(xs), np.amax(xs))
    
    data_bins = [ys[(xs >= x0s[i]) & (xs <= x1s[i])] for i in range(len(x0s))]
    data_xs = [xs[(xs >= x0s[i]) & (xs <= x1s[i])] for i in range(len(x0s))]
    
    y = [np.nanmedian(data) for data in data_bins]
    s = [np.nanstd(data) for data in data_bins]
    x = 0.5*(x0s+x1s)
    x = [np.nanmedian(xvals) for xvals in data_xs]
    x = [np.nanmean(xvals) for xvals in data_xs]
    
    return np.array(x), np.array(y), np.array(s)
    