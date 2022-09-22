#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:23:08 2021

@author: kris
"""

import numpy as np
import pickle
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from RNN_utils import sim_rnn, gen_obs, run_cca
from utils import global_params
np.random.seed(8180909)

iid_rnns = pickle.load(open('./results/rnn/reps/rep0_data.pickled', 'rb'))
params = pickle.load(open('./results/rnn/reps/rep0_params.pickled', 'rb'))

N = params['N']
ntrial = 100
scale_rate = 0.05
scale_rate = 0.025
ex_ns = np.arange(N)
### simulate both RNNs

x_ics, Ws, w_outs, bs, x_outs, all_xs = iid_rnns

print('generating trials')
x1, y1 = sim_rnn(x_ics[0], Ws[0], w_outs[0], bs[0], params, batch = ntrial)
xr, yr = sim_rnn(x_ics[0], Ws[0], w_outs[0], bs[0], params, batch = ntrial)
x2, y2 = sim_rnn(x_ics[1], Ws[1], w_outs[1], bs[1], params, batch = ntrial)
meanx = np.mean([np.mean(x) for x in [x1, x2, xr]])
x1, xr, x2 = [x*meanx/np.mean(x) for x in [x1, xr, x2]]
x1 = np.maximum(x1, 0)
x2 = np.maximum(x2, 0)
xr = np.maximum(xr, 0)

peth1, spikes1, spikes_c1, vel1, ts = gen_obs(x1, y1, scale_rate = scale_rate)
pethr, spikesr, spikes_cr, velr, ts = gen_obs(xr, yr, scale_rate = scale_rate)
peth2, spikes2, spikes_c2, vel2, ts = gen_obs(x2, y2, scale_rate = scale_rate)

print('computing correlations')
def comp_rs(p1, p2):
    rs = [pearsonr(p1[n, :], p2[n, :])[0] for n in range(N) if min(sum(p1[n, :]), sum(p2[n, :])) >= global_params['minspike']]
    return np.array(rs)

### Compute correlations ###
r_same = comp_rs(peth1, pethr)
r_diff = comp_rs(peth1, peth2)

rs = np.linspace(-1, 1, 501)
h_same, h_diff = [np.histogram(r, bins = rs)[0].astype(float) for r in [r_same, r_diff]]
h_same, h_diff = [h/np.sum(h)/(rs[1]-rs[0]) for h in [h_same, h_diff]]
c_same, c_diff = [gaussian_filter1d(h, 10, mode = 'nearest') for h in [h_same, h_diff]]

### compute CCs ###
print('computing CCs')
nrep = 250
NCC = 50
CCs = np.zeros((nrep, 2))
for n in range(nrep):
    inds = np.random.choice(N, NCC, replace = False)
    
    a1, a2, ar = [s[:, inds, :].transpose(1, 0, 2).reshape(NCC, -1) for s in [spikes_c1, spikes_c2, spikes_cr]]
    
    CCs[n, 0] = run_cca(a1, a2)
    CCs[n, 1] = run_cca(a1, ar)
    
h_same, h_diff = [np.histogram(CCs[:, i], bins = rs)[0].astype(float) for i in range(2)]
h_same, h_diff = [h/np.sum(h)/(rs[1]-rs[0]) for h in [h_same, h_diff]]
c_same, c_diff = [gaussian_filter1d(h, 10) for h in [h_same, h_diff]]

### store result (correlations and activity) ###

data = {}
data['CCs'] = CCs
data['r_same'] = r_same
data['r_diff'] = r_diff
data['ex_peth'] = peth1[ex_ns, :]
data['ex_spikes'] = spikes1[:, ex_ns, :]
data['ex_ns'] = ex_ns
data['y'] = y1


print('\nsaving data')
pickle.dump(data, open('./results/rnn/iid_analyses.p', 'wb'))
