#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:04:26 2020

@author: kris
"""

import numpy as np
import sys
import scipy.stats
import copy
import pickle

from RNN_utils import sim_rnn_single, interp_mats
from sklearn.linear_model import LinearRegression


from RNN_train_loop import train_RNN
import tensorflow as tf

reps = np.arange(10)
#reps = [0]

for rep in reps:

    print('\nnew repetition:', rep)
    
    seed = 12581903+rep
    np.random.seed(seed)
    tf.random.set_seed(seed)


    #%% initialize parameters and target functions
    N = 250; T = 250; dt = 1; n_steps = int(T/dt); tau = 10; n_train_steps = 1001; noise = 2e-1; scale_noise = 1.0; reg = 1e-4
    lrate = 5e-4
    nout = 5

    params = {'N': N, 'T': T, 'tau': tau, 'noise': noise, 'dt': dt}

    ts = np.arange(T)

    ell = T/6.
    K = np.exp(-(ts[:, None] - ts[None, :])**2 / (2*ell**2))
    L = np.linalg.cholesky(K+np.eye(T)*1e-6)
    y_np = (L @ np.random.normal(0, 1, (T, nout))).T

    print('simulating network with', N, 'recurrent neurons and', nout, 'readout neurons.\n\n')

    #%% train on target

    from_pickled = False

    if from_pickled:
        x_ics, Ws, w_outs, bs, x_outs, all_xs = pickle.load(open('./results/data.pickled', 'rb'))

    else:
        x_ics, Ws, w_outs, bs, x_outs, all_xs = [[]for i in range(6)]
        for i in range(2):

            new_train_steps = n_train_steps#+i*4000

            #draw random initial weights and states
            b_np = np.random.normal(0, 1, (nout, 1) ) # 1 x nout
            x_ic_np = np.random.uniform(0.5, 1, (N, 1) ) # N x 1
            W_np = np.random.normal(0, 0.8/np.sqrt(N), (N, N) ) # N x N
            w_out_np = np.random.normal(0, 1/N, (nout, N)) # nout x N

            #simulate initial network
            xs_i, x_out_i = sim_rnn_single(x_ic_np, W_np, w_out_np, b_np, noise = 0, T=T, tau=tau)

            #train recurrent network
            cost_over_epochs, x_ic, W, w_out, b, W_sims, w_out_sims = train_RNN(x_ic_np, b_np, W_np, w_out_np,
                                                                    y_np, new_train_steps,
                                                                    dt, tau, noise, reg = reg,
                                                                    lrate = lrate)
            #simulate trained network and plot
            xs, x_out = sim_rnn_single(x_ic, W, w_out, b, noise = scale_noise*noise, T=T, tau=tau)

            x_ics.append(x_ic); Ws.append(W); w_outs.append(w_out)
            bs.append(b); x_outs.append(x_out); all_xs.append(xs)

        #simply store the parameters and then we can always re-simulate the networks
        pickle.dump([x_ics, Ws, w_outs, bs, x_outs, all_xs], open('results/rnn/reps/rep'+str(rep)+'_data.pickled', 'wb'))
        pickle.dump(params, open('results/rnn/reps/rep'+str(rep)+'_params.pickled', 'wb'))
        pickle.dump(y_np, open('results/rnn/reps/rep'+str(rep)+'_target.pickled', 'wb'))



    randW = scipy.stats.pearsonr(Ws[0].reshape(N**2), Ws[1].reshape(N**2))


    ''
    #%% interpolate weight matrices - only use half the training steps since starting from a good place

    np.random.seed(seed)
    tf.random.set_seed(seed)
    print('interpolating weight matrices')

    frac = 0.70 #fraction of second matrix to mix in
    ninterp = 7

    interp_steps = n_train_steps
    interp_steps = int(round(n_train_steps/2))
    interp_lr = lrate/2

    W_interp, x_ic_interp, w_out_interp, b_interp = [[l[0], frac*l[1]+(1-frac)*l[0]] for l in [Ws, x_ics, w_outs, bs]]
    x_ic_interp, W_interp, w_out_interp, b_interp, x_out_interp, xs_interp = interp_mats(
            W_interp, x_ic_interp, w_out_interp, b_interp, y_np, interp_steps, dt, tau, noise, ninterp = ninterp, lrate = interp_lr, reg = reg)

    pickle.dump([x_ic_interp, W_interp, w_out_interp, b_interp, x_out_interp, xs_interp],
                open('results/rnn/reps/rep'+str(rep)+'_data_interp.pickled', 'wb'))


    ### repeat with self-interpolation ##
    W_interp, x_ic_interp, w_out_interp, b_interp = [[l[0], l[0]] for l in [Ws, x_ics, w_outs, bs]]
    x_ic_interp, W_interp, w_out_interp, b_interp, x_out_interp, xs_interp = interp_mats(
            W_interp, x_ic_interp, w_out_interp, b_interp, y_np, interp_steps, dt, tau, noise, ninterp = ninterp, lrate = interp_lr, reg = reg)

    pickle.dump([x_ic_interp, W_interp, w_out_interp, b_interp, x_out_interp, xs_interp],
                open('results/rnn/reps/rep'+str(rep)+'_data_sinterp.pickled', 'wb'))

