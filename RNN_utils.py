#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:43:03 2020

@author: kris
"""
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from RNN_train_loop import train_RNN

def IO_fun_np(x_in):
    return np.maximum(0., x_in)

def sim_rnn(x_ic_f, W_f, w_out_f, b_f, params, batch = 100):
    """
    return
    activity x \in (batch, N, T)
    output y \in (batch, nout, T)
    """    
    N, noise, T, dt, tau = [params[k] for k in ['N', 'noise', 'T', 'dt', 'tau']]
    
    x_f = np.zeros( (N, batch, T) )
    x_f[..., 0] = np.maximum(0., x_ic_f)
    
    for t in range(T-1):
        inp_f = W_f @ x_f[..., t] #N x batch
        inp_f = inp_f + np.random.normal(0, noise, size = (N, batch))/np.sqrt(dt)
        x = x_f[..., t] + (dt/tau) * (-x_f[..., t] + inp_f )
        x_f[..., t+1] = IO_fun_np(x)
        
    x_f = x_f.transpose(1, 0, 2) # batch x N x T
    y_f = w_out_f @ x_f + b_f
    
    return x_f, y_f


def sim_rnn_single(x_ic_f, W_f, w_out_f, b_f, noise=2, T=1000, tau=50, dt = 1, tau_noise = 10):
    N = len(x_ic_f)
    x_f = np.zeros( (N, T) )
    x_f[:, 0] = np.maximum(0., x_ic_f[:,0])
    for t in range(T-1):
        inp_f = W_f @ x_f[:, t]
        inp_f = inp_f + np.random.normal(0, noise, size = N)/np.sqrt(dt)
        x = x_f[:, t] + (dt/tau) * (-x_f[:, t] + inp_f )
        x_f[:, t+1] = IO_fun_np(x)
    
    return x_f, w_out_f @ x_f + b_f


def gen_obs(x, y, kernel = 15, scale_rate = 0.1, max_fr = None):
    '''
    x \in (batch, N, T): activity
    y \in (batch, N, T): output
    '''
    
    batch, N, T = x.shape
    x = x*scale_rate
    
    x = x * scale_rate/np.mean(x)
    
    if max_fr is not None:
        fr = np.amax(np.mean(x, axis = 0), axis = -1)*x.shape[-1] #maximum spike/second
        new_fr = max_fr - np.log(1+np.exp(max_fr-fr)) #soft saturation
        new_fr = np.maximum(1e-6, new_fr)
        fr = np.maximum(1e-6, fr)
        x = x*new_fr[None, :, None]/fr[None, :, None]
    
    spikes = np.random.poisson(x) #batch, N, T
    peth = np.sum(spikes, axis = 0) #sum across trials; N x T
    ## convolve with Gaussian filter
    peth_c = gaussian_filter1d(peth.astype(float), kernel * T/1000, mode = 'nearest', axis = -1)
    spikes_c = gaussian_filter1d(spikes.astype(float), 2*kernel * T/1000, mode = 'nearest', axis = -1)
    
    vel = np.mean(y, axis = 0)
    ts = np.arange(T)
    
    inds = np.arange(12, 238)
    peth_c, spikes, spikes_c, vel, ts = [arr[..., inds] for arr in [peth_c, spikes, spikes_c, vel, ts]]
    
    return peth_c, spikes, spikes_c, vel, ts
    
def run_cca(Xa, Xb, nPC = 10, nCC = 4, return_aligned = False):
    '''
    Run CCA to align Xb to Xa
    X is n by T where n is #units, T is number of datapoints (timepoints by trials)
    nPC is the number of PCA dimensions to retain
    nCC is the number of CCA dimensions to compute similarity over
    '''
    
    #dimensionality reduction by PCA
    pca = PCA(n_components = nPC) #define model
    La = pca.fit_transform(Xa.T).T #array that is nPC x T
    Lb = pca.fit_transform(Xb.T).T #array that is nPC x T

    #QR decomposition of both latent matrices
    Qa, Ra = np.linalg.qr(La.T, mode = 'reduced') #T x nPC, nPC x nPC
    Qb, Rb = np.linalg.qr(Lb.T, mode = 'reduced') #T x nPC, nPC x nPC
    
    
    QaTQb = Qa.T @ Qb # nPC x nPC
    
    #SVD
    U, S, Vh = np.linalg.svd( QaTQb, full_matrices=True) # nPC x nPC, nPC x nPC (diagonal), nPC x nPC
    cor_CC = np.mean(S[:nCC])

    if return_aligned:
        #PCA correlation; average across dimensions
        cor_PC = np.mean([pearsonr(La[i, :], Lb[i, :])[0] for i in range(nPC)])
        #align dimensions
        Ma = np.linalg.inv(Ra) @ U # nPC x nPC
        Mb = np.linalg.inv(Rb) @ Vh.T #nPC x nPC
        Lb_aligned = (Lb.T @ Mb @ np.linalg.inv(Ma)).T # nPC x T
        
        return La, Lb, Lb_aligned, cor_PC, cor_CC
    
    else:
        #don't return the aligned trajectories
        return cor_CC
    
        
def interp_mats(Ws, x_ics, w_outs, bs, y_np, n_train_steps, dt, tau, noise, reg = 1e-3, ninterp = 5, lrate = 50e-6):
    '''
    Interpolate between two RNNs
    '''

    nout, T = y_np.shape
    N = Ws[0].shape[0]
    
    fracs = np.linspace(0, 1, ninterp)
    W_interp = [ (1-frac)*Ws[0] + (frac)*Ws[1] for frac in fracs ]
    x_ic_interp = [ (1-frac)*x_ics[0] + (frac)*x_ics[1] for frac in fracs ]
    w_out_interp = [ (1-frac)*w_outs[0] + (frac)*w_outs[1] for frac in fracs ]
    b_interp = [ (1-frac)*bs[0] + (frac)*bs[1] for frac in fracs ]
    xs_interp = []
    x_out_interp = []
    for i, W in enumerate(W_interp):
        print('\ninterpolation', i, 'of', ninterp)
        #train from interpolated data, fixing readout weights
        _, x_ic, W, w_out, b, _, _ = train_RNN(x_ic_interp[i],
                                               b_interp[i],
                                               W_interp[i],
                                               w_out_interp[i],
                                               y_np, n_train_steps,
                                               dt, tau, noise,
                                               reg = reg,
                                               readout = 'fixed', lrate = lrate)
        x_ic_interp[i] = x_ic
        b_interp[i] = b
        W_interp[i] = W
        w_out_interp[i] = w_out
        
        #simulate netwoorks
        xs, x_out = sim_rnn_single(x_ic, W, w_out, b, noise = 0, T=T, tau=tau)
        xs_interp.append(xs)
        x_out_interp.append(x_out)
    
    return [x_ic_interp, W_interp, w_out_interp, b_interp, x_out_interp, xs_interp]
    
