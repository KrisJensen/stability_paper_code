#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:23:22 2021

@author: kris
"""

import numpy as np
import pickle
from scipy.stats import pearsonr, binned_statistic
from scipy.ndimage import gaussian_filter1d
from RNN_utils import sim_rnn, gen_obs, run_cca
from compute_neural_sim import calc_alphas
from utils import global_params
import sys

reps = np.arange(10)
data = {}

if len(sys.argv) > 1:
    drifting = bool(int(sys.argv[1]))
else:
    drifting = True
print('drifting:', drifting)

for rep in reps:

    np.random.seed(8180909)
    data[rep] = {}

    if drifting:
        iid_rnns = pickle.load(open('./results/rnn/reps/rep'+str(rep)+'_data_interp.pickled', 'rb'))
        driftstr = '_drift'
    else:
        iid_rnns = pickle.load(open('./results/rnn/reps/rep'+str(rep)+'_data_sinterp.pickled', 'rb'))
        driftstr = '_stable'

    params = pickle.load(open('./results/rnn/reps/rep'+str(rep)+'_params.pickled', 'rb'))

    N, T = [params[k] for k in ['N', 'T']]
    nout = 5
    ntrials = 100
    scale_rate = 0.05
    scale_rate = 0.025
    max_fr = None
    ex_ns = np.arange(50)

    x_ics, Ws, w_outs, bs, x_outs, all_xs = iid_rnns

    ninterp = len(x_ics)
    xs = np.zeros((ninterp, ntrials, N, T))
    ys = np.zeros((ninterp, ntrials, nout, T))
    peths = [] #peths
    vels = [] #mean velocities
    spikes = [] #raw spikes
    spikes_c = [] #convolved with Gaussian
    print('\ngenerating trials')
    for i in range(ninterp):
        x, y = sim_rnn(x_ics[i], Ws[i], w_outs[i], bs[i], params, batch = ntrials)
        xs[i, ...] = x
        ys[i, ...] = y

    xs = xs * np.mean(xs) / np.mean(xs, axis = (1,2,3), keepdims = True)
    for i in range(ninterp):
        peth, spike, spike_c, vel, ts = gen_obs(np.maximum(xs[i], 0), ys[i], scale_rate = scale_rate, max_fr = max_fr)
        peths.append(peth)
        vels.append(vel)
        spikes.append(spike)
        spikes_c.append(spike_c)

    peths, vels, spikes, spikes_c = np.array(peths), np.array(vels), np.array(spikes), np.array(spikes_c)

    ### compute behavioral similarity
    print('\ncomputing behavioral similarity')
    sims, dts = [[[] for _ in range(nout)] for _ in range(2)]
    for i1 in range(ninterp):
        v1 = vels[i1, ...]
        for i2 in range(i1+1, ninterp):
            v2 = vels[i2, ...]
            for n in range(nout):
                sims[n].append(pearsonr(v1[n], v2[n])[0])
                dts[n].append(float(i2-i1))

    data[rep]['behav_sims'] = sims
    data[rep]['behav_dts'] = dts
    bins = np.arange(0.5, 7.5, 1)
    bin_dts = 0.5*(bins[1:] + bins[:-1])
    bin_sims = np.zeros((nout, len(bin_dts)))*np.nan
    for n in range(nout):
        dt, sim = dts[n], sims[n]
        bin_sims[n, :] = binned_statistic(dt, sim, bins = bins)[0]

    m, s = np.nanmean(bin_sims, axis = 0), np.nanstd(bin_sims, axis = 0)
    s = s / np.sqrt(np.sum(1-np.isnan(bin_sims), axis = 0))

    ### plot neural similarity over time
    print('\ncomputing neural similarity')
    sims, dts = [[[] for _ in range(N)] for _ in range(2)]
    for i1 in range(ninterp):
        p1 = peths[i1, ...]
        for i2 in range(i1+1, ninterp):
            p2 = peths[i2, ...]
            for n in range(N):
                if min(sum(p1[n]), sum(p2[n])) >= global_params['minspike']:
                    sims[n].append(pearsonr(p1[n], p2[n])[0])
                    dts[n].append(float(i2-i1))

    data[rep]['neural_sims'] = sims
    data[rep]['neural_dts'] = dts
    bins = np.arange(0.5, 7.5, 1)
    bin_dts = 0.5*(bins[1:] + bins[:-1])
    bin_sims = np.zeros((N, len(bin_dts)))*np.nan            
    for n in range(N):
        dt, sim = dts[n], sims[n]
        if len(dt) >= 1:
            bin_sims[n, :] = binned_statistic(dt, sim, bins = bins)[0]

    m, s = np.nanmean(bin_sims, axis = 0), np.nanstd(bin_sims, axis = 0)
    s = s / np.sqrt(np.sum(1-np.isnan(bin_sims), axis = 0))

    ### compute stability indices ###
    alphas, _ = calc_alphas(np.arange(N), dts, sims, comp_error = False)
    data[rep]['alphas'] = alphas
        
    if rep == 0:
        data[rep]['example_spikes'] = spikes[:, :, ex_ns, :]
        data[rep]['ex_ns'] = ex_ns

    ### plot CCA similarity over time
    print('\ncomputing CCA similarity')
    NCC = 50
    ngroup = int(N/NCC)
    sims, dts = [[[] for _ in range(ngroup)] for _ in range(2)]
    for i1 in range(ninterp):
        s1 = spikes_c[i1, ...]
        for i2 in range(i1+1, ninterp):
            s2 = spikes_c[i2, ...]
            for n in range(ngroup):
                a1, a2 = [s[:, n*NCC:(n+1)*NCC, :].transpose(1, 0, 2).reshape(NCC, -1) for s in [s1, s2]]
                sims[n].append(run_cca(a1, a2))
                dts[n].append(float(i2-i1))
    data[rep]['CCA_sims'] = sims
    data[rep]['CCA_dts'] = dts
    bins = np.arange(0.5, 7.5, 1)
    bin_dts = 0.5*(bins[1:] + bins[:-1])
    bin_sims = np.zeros((ngroup, len(bin_dts)))*np.nan            
    for n in range(ngroup):
        dt, sim = dts[n], sims[n]
        bin_sims[n, :] = binned_statistic(dt, sim, bins = bins)[0]

    m, s = np.nanmean(bin_sims, axis = 0), np.nanstd(bin_sims, axis = 0)
    s = s / np.sqrt(np.sum(1-np.isnan(bin_sims), axis = 0))
    
    if rep == 0:
        print('\nsaving data')
        pickle.dump(data, open('./results/rnn/interp_analyses'+driftstr+'.p', 'wb'))
        
    print('finished rep', rep)

print('\nsaving data')
pickle.dump(data, open('./results/rnn/interp_analyses_rep'+driftstr+'.p', 'wb'))
