#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:39:19 2020

@author: kris
"""
import numpy as np
import scipy.stats
import time
import tensorflow as tf
import importlib

####Train RNN

#%% run training procedure

def train_RNN(x_ic_np, b_np, W_np, w_out_np, y_np, n_train_steps, dt, tau, noise, reg = 1e-3,
              readout = 'learned', recurrent = 'learned', ics = 'learned', bias = 'learned', lrate = 50e-6,
             batch = 20):
    '''
    simple recurrent network with N densely connected recurrent units and nout readout units
    train network such that each readout unit produces a specific trajectory
    
    x_ic_np: initial 'initial conditions' as numpy array
    b_np: initial biases
    W_np: initial recurrent weights
    w_out_np: initial readout weights
    y_np: target array (shape: nout x ntimesteps)
    noise is currently not temporally correlated; fix this at some point
    if discount: only impose constraints on a subset of the full simulation
    readout weights are 'learned' or 'fixed' '''
    
    importlib.reload(tf)
    
    
    nout, N = w_out_np.shape #number of output and recurrent units
    T = y_np.shape[1] #number of timesteps

    #initialize tf variables
    
    tf_vars = []
    
    if ics == 'learned':
        x_ic = tf.Variable(x_ic_np, dtype = tf.float32)
        tf_vars.append(x_ic)
    else:
        x_ic = tf.constant(x_ic_np, dtype = tf.float32)
    
    if recurrent == 'learned':
        W = tf.Variable(W_np, dtype = tf.float32)
        tf_vars.append(W)
    else:
        W = tf.constant(W_np, dtype = tf.float32)
        
    if bias == 'learned':
        b = tf.Variable(b_np, dtype = tf.float32)
        tf_vars.append(b)
    else:
        b = tf.constant(b_np, dtype = tf.float32)
        
    if readout == 'learned':
        w_out = tf.Variable(w_out_np, dtype = tf.float32)
        tf_vars.append(w_out)
    else:
        w_out = tf.constant(w_out_np, dtype = tf.float32)
    
    #initialize tf constants
    y = tf.constant(y_np, dtype = tf.float32)
    
    #%% define function forward
    
    @tf.function
    def calc_loss(y, t, x_out):
        l1 = tf.reduce_sum( tf.square(y[:, t][:, None] - x_out) )/batch
        return l1
    
    
    optimizer = tf.optimizers.Adam(learning_rate = lrate) #use Adam
    @tf.function
    def forward():
        with tf.GradientTape() as tape:
            x = np.ones((x_ic.shape[0], batch))*tf.nn.relu(x_ic)
            loss = tf.constant(0.0, dtype = tf.float32)
            loss = loss + calc_loss( y, 0, (w_out@x+b) )
            for t in tf.range(1, T):
                inp = W @ x #deterministic update
                inp = inp + tf.random.normal((N,batch), 0, noise, dtype = tf.float32)/np.sqrt(dt)
                x = x + (dt/tau) * (-x + inp) #update state
                x = tf.nn.relu(x)
                
                loss = loss + calc_loss( y, t, (w_out@x+b))
                
            l2 = reg*tf.reduce_sum( tf.square(W) )
            l3 = reg*tf.reduce_sum( tf.square(w_out) )
            loss = loss + l2 + l3
        gradients = tape.gradient(loss, tf_vars)
        optimizer.apply_gradients(zip(gradients, tf_vars)) #if you want to optimize
        return float(loss)
    
    
    #%% Do grad descent
    t = time.time()
    
    cost_over_epochs, W_sims, w_out_sims = [np.zeros(n_train_steps) for i in range(3)] #is this a bad way to store weights etc?
    for epoch in range(n_train_steps):
        loss = forward()
        cost_over_epochs[epoch] = loss

        if epoch % 100 == 0:
            W_f = np.array([np.array(W[i]) for i in range(N)])
            w_out_f = np.array([np.array(w_out[i]) for i in range(nout)])
            W_sims[epoch] = scipy.stats.pearsonr(W_f.reshape(N**2), W_np.reshape(N**2))[0]
            w_out_sims[epoch] = scipy.stats.pearsonr(w_out_f.reshape(N*nout), w_out_np.reshape(N*nout))[0]
            print(epoch, cost_over_epochs[epoch]/(nout*T), W_sims[epoch], w_out_sims[epoch], time.time()-t) #print progress
            t = time.time()
    
    W_f = np.array([np.array(W[i]) for i in range(N)])
    x_ic_f = np.array([np.array(x_ic[i]) for i in range(N)])
    b_f = np.array([np.array(b[i]) for i in range(nout)])
    w_out_f = np.array([np.array(w_out[i]) for i in range(nout)])
    
    return cost_over_epochs, x_ic_f, W_f, w_out_f, b_f, W_sims, w_out_sims

