#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:01:19 2021

@author: kris
"""


import numpy as np
import pickle
from scipy.interpolate import CubicSpline
import time
from scipy.ndimage import gaussian_filter1d
import sys
from utils import global_params, get_modes

def nan_helper(y):
    """
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def warp_twotap_behav(rat, day, limb, keep, trim = False):
    
    ###fit splines###
    ncoord = rat['trials'][day]['kinematics'][limb].shape[-1]
    times = rat['trials'][day]['times']
    splines = [[CubicSpline(times, rat['trials'][day]['kinematics'][limb][i1, :, i2]) for i2 in range(ncoord)]for i1 in range(len(keep))] #ntrial x 2
    
    ### warp behavior ###
    for k in ['kinematics_w', 'vels_w', 'acc_w']:
        rat['trials'][day][k][limb] = np.zeros(rat['trials'][day]['kinematics'][limb].shape)
    
    tint = 0.7 #target interval (seconds)
    ipis = rat['trials'][day]['ipis']
    
    for itrial in np.where(keep)[0]:
        Int = ipis[itrial]
        if trim:
            Int = tint #no warping

        t1 = (np.arange(61) - 60) / 120 #timepoints in seconds before first tap
        t2 = (np.arange(61, 145) - 60) / 120 * Int / tint #between taps
        t3 = (np.arange(145, 241) - 60) / 120 + Int - tint #after second tap
        newtimes = np.concatenate([t1, t2, t3])
        
        for icoord in range(2):                    
            rat['trials'][day]['kinematics_w'][limb][itrial, :, icoord] = splines[itrial][icoord](newtimes)
            rat['trials'][day]['vels_w'][limb][itrial, :, icoord] = splines[itrial][icoord](newtimes, 1) #first derivative
            rat['trials'][day]['acc_w'][limb][itrial, :, icoord] = splines[itrial][icoord](newtimes, 2) #first derivative
    return
    
def warp_wds_behav(rat, day, limb, keep, tmin = -0.2, tmax = 0.5, trim = False):
    
    acc = rat['trials'][day]['kinematics'][limb]
    times = rat['trials'][day]['times']
    splines = [[CubicSpline(times, acc[i1, :, i2]) for i2 in range(3)]for i1 in range(len(keep))]

    tint =rat['targetint'] #target interval
    periods = rat['trials'][day]['periods']
    
    ### warp behavior ###
    times = np.arange(tmin, tmax, 1/300) #times we want to extract
    for k in ['kinematics_w', 'vels_w', 'acc_w']:
        shape0 = rat['trials'][day]['kinematics'][limb].shape
        rat['trials'][day][k][limb] = np.zeros((shape0[0], len(times), shape0[2]))
    
    for itrial in np.where(keep)[0]:
        event = rat['trials'][day]['events'][itrial]
        extimes = np.concatenate([rat['trials'][day]['peaktimes'][itrial][2], rat['trials'][day]['valtimes'][itrial][2]])
        extimes = np.concatenate([rat['trials'][day]['peaktimes'][itrial][2]])
    
        period = periods[itrial]
        t0, t2 = np.sort(extimes-event)[[0, -1]]+np.array([-1, 1])*period/4 #first and last peak/valley
        factor = period/tint

        if trim:
            factor = 1 #no warping
        
        tt1 = times[times < t0] + t0*(factor-1)
        tt2 = times[(times >= t0) & (times <= t2)] * factor
        tt3 = times[times > t2] + t2*(factor-1)
        newtimes = np.concatenate([tt1, tt2, tt3])
    
        for icoord in range(3):                    
            for k in ['kinematics_w', 'vels_w', 'acc_w']: #use acceleration for all of it
                rat['trials'][day][k][limb][itrial, :, icoord] = splines[itrial][icoord](newtimes)
    return

def parse_behav(rat, maxnan = 0.5, trim = False):
    '''interpolate kinematics, fit spline, and extract time warped position+velocity'''
    all_keep = {}
    for day in rat['trials'].keys():
        for k in ['kinematics_w', 'vels_w', 'acc_w']:
            rat['trials'][day][k] = {}  
        
        for limb in rat['trials'][day]['kinematics'].keys():
            ### interpolate behavior ###
            ncoord = rat['trials'][day]['kinematics'][limb].shape[-1]
            nanfracs = np.mean(np.isnan(rat['trials'][day]['kinematics'][limb]), axis = 1) #trials x 2
            nanfracs = (nanfracs <= maxnan) #keep
            keep = np.prod(nanfracs, axis = -1).astype(bool) #'and'
            all_keep[day] = keep
                
            for itrial in np.arange(len(keep)):
                #interpolate through nans
                for icoord in range(ncoord):
                    nans, x = nan_helper(rat['trials'][day]['kinematics'][limb][itrial, :,icoord])
                    rat['trials'][day]['kinematics'][limb][itrial,nans,icoord] = np.interp(x(nans), x(~nans), rat['trials'][day]['kinematics'][limb][itrial,~nans,icoord])
            if 'wds' in rat['name']:
                warp_wds_behav(rat, day, limb, keep, trim = trim) #in-place
            else:
                warp_twotap_behav(rat, day, limb, keep, trim = trim) #in-place
        
    return rat, all_keep


def warp_twotap_activity(rat, all_keep, tmin, tmax, tint = 'default', trim = False):
    '''warp neural activity for all units'''
    
    tint = 0.7
    for unum in rat['units'].keys():
        for day in rat['units'][unum].keys():
            keep = all_keep[day]
            
            ipis = rat['trials'][day]['ipis']
            raster_w = []
            
            for itrial in range(len(keep)):
                Int = ipis[itrial]

                if trim:
                    Int = tint #no warping
                
                spikes = rat['units'][unum][day]['raster'][itrial]
            
                start = 0
                end = len(spikes)
                
                #now do warped
                newspikes = spikes[start:end]
                inds1 = np.where(newspikes < 0)[0]
                inds2 = np.where((newspikes >= 0) & (newspikes <= Int))[0]
                inds3 = np.where(newspikes > Int)[0]
                newspikes = np.concatenate((newspikes[inds1],
                                     newspikes[inds2]*tint/Int,
                                     newspikes[inds3]+(tint-Int) ))
            
                raster_w.append(newspikes)
            rat['units'][unum][day]['raster_w'] = raster_w
    return rat

def warp_wds_activity(rat, all_keep, tmin, tmax, tint = 'default', trim = False):
    
    tint = rat['targetint']
    for unum in rat['units'].keys():
        for day in rat['units'][unum].keys():
            keep = all_keep[day]
            
            periods = rat['trials'][day]['periods']
            raster_w = []
    
            for itrial in range(len(keep)):
                event = rat['trials'][day]['events'][itrial]
                extimes = np.concatenate([rat['trials'][day]['peaktimes'][itrial][2], rat['trials'][day]['valtimes'][itrial][2]])
            
                period = periods[itrial]
                t0, t2 = np.sort(extimes-event)[[0, -1]]+np.array([-1, 1])*period/4 #first and last peak/valley
                factor = period/tint

                if trim:
                    factor = 1 #no warping!
                
                spikes = rat['units'][unum][day]['raster'][itrial]
            
                start = 0
                end = len(spikes)
                
                newspikes = spikes[start:end]
                inds1 = np.where(newspikes < t0)[0]
                inds2 = np.where((newspikes >= t0) & (newspikes <= t2))[0]
                inds3 = np.where(newspikes > t2)[0]
                newspikes = np.concatenate((newspikes[inds1]+t0*(factor-1),
                                     newspikes[inds2]*factor,
                                     newspikes[inds3]+t2*(factor-1) ))

            
                raster_w.append(newspikes)
            rat['units'][unum][day]['raster_w'] = raster_w

    return rat

def select_trials(rat, all_keep):
    '''only keep trials in 'all_keep' for each day
    allows us to e.g. discard trials with occlusion'''
    
    for day in all_keep.keys():
        tokeep = all_keep[day]
        for k in ['ipis', 'inds', 'modes', 'events']:
            rat['trials'][day][k] = rat['trials'][day][k][tokeep]
        for kins in ['kinematics', 'kinematics_w', 'vels_w', 'acc_w']:
            for k in rat['trials'][day][kins].keys():
                rat['trials'][day][kins][k] = rat['trials'][day][kins][k][tokeep, ...]
                       
    for unum in rat['units'].keys():
        for day in rat['units'][unum].keys():
            tokeep = all_keep[day]
            for k in ['raster', 'raster_w']:
                raster = rat['units'][unum][day][k]
                rat['units'][unum][day][k] = [raster[i] for i in range(len(tokeep)) if tokeep[i]]
                assert len(rat['units'][unum][day][k]) == len(rat['trials'][day]['ipis'])
    return rat
    
def calc_peth(raster, tmin, tmax, wds = False, gauss = True):
    
    binsize = global_params['binsize']
    bins = np.arange(tmin, tmax+binsize, binsize)
    c_res = 0.001
    if global_params['smooth_peth'] and gauss:
        ### gaussian filter ###
        peth_new = np.histogram(raster, bins = np.arange(tmin-binsize, tmax+binsize+c_res, c_res))[0].astype(float)
        res = int(binsize/c_res) #20 ms
        
        sig = 1.5*res/2.
        conv_key = 'conv_wds' if wds else 'conv_2tap'
        sig = float(res) * global_params[conv_key] / binsize
        
        peth_c = gaussian_filter1d(peth_new, sig, mode = 'nearest') #3/4 binsize
        peth_c = peth_c[int(1.5*res):len(peth_c)-res:res]*res
        peth = peth_c
        
    else:
        ### histogram ###
        peth = np.histogram(raster, bins = bins)[0].astype(float)
            
    return peth

def parse(rat, subselect = True, maxnan = 0.5, tmax = 'default', tmin = 'default', trim = False):
    '''
    compute raster and perform time warping
    '''
    
    if tmin == 'default':
        tmin = -0.2 if ('wds' in rat['name']) else -0.1
    if tmax == 'default':
        tmax = 0.5 if ('wds' in rat['name']) else 0.8
    
    if subselect:
        rat = subselect_trials(rat)
        
    ### warp behavior and compute velocity ###
    rat, all_keep = parse_behav(rat, maxnan = maxnan, trim = trim)
    
    ### warp neural activity ###
    if 'wds' in rat['name']:
        wds = True
        rat = warp_wds_activity(rat, all_keep, tmin, tmax, trim = trim)
    else:
        wds = False
        rat = warp_twotap_activity(rat, all_keep, tmin, tmax, trim = trim)
    
    ### only keep 'tokeep' ###
    rat = select_trials(rat, all_keep)
    
    ### compute peths ###
    for unum in rat['units'].keys():
        for day in rat['units'][unum].keys():
            raster_w = rat['units'][unum][day]['raster_w']
            peth_t = np.array([calc_peth(raster, tmin, tmax, wds = wds, gauss = False) for raster in raster_w])
            
            raster_w = np.concatenate(raster_w)
            peth = calc_peth(raster_w, tmin, tmax, wds = wds)
           
            rat['units'][unum][day]['peth_w'] = peth
            rat['units'][unum][day]['peth_w_t'] = peth_t # by trial
    
    return rat
    
#%%
def subselect_trials(rat, mode = 'default', ipi_min = 'default', ipi_max = 'default', mintrial = 10):
    ### mintrial is the minimum number of trials to construct a peth
    if ipi_min == 'default':
        ipi_min = -np.inf if 'wds' in rat['name'] else 0.6
    if ipi_max == 'default':
        ipi_max = np.inf if 'wds' in rat['name'] else 0.8
    
    if mode == 'default':
        mode = get_modes(rat['name'])[0]
    
    ### find suitable trials and subselect behavior ###
    all_inds = {}
    todel = []
    for day in rat['trials'].keys():
        inds = (rat['trials'][day]['ipis'] <= ipi_max) & (rat['trials'][day]['ipis'] >= ipi_min) & (rat['trials'][day]['modes'] == mode)
        
        if 'wds' in rat['name']:
            Ls = np.array([acc.shape[0] for acc in rat['trials'][day]['kinematics']['acc']])
            inds = inds & (Ls == 450)
            kinematics = {'acc': np.array([rat['trials'][day]['kinematics']['acc'][ind] for ind in np.where(inds)[0]])}
        else:
            kinematics = {k: rat['trials'][day]['kinematics'][k][inds, ...] for k in rat['trials'][day]['kinematics'].keys()}
        
        all_inds[day] = np.where(inds)[0]
            
        times = rat['trials'][day]['times']
        if 'wds' in rat['name']:
            peaktimes = [rat['trials'][day]['peaktimes'][ind] for ind in np.where(inds)[0]]
            valtimes = [rat['trials'][day]['valtimes'][ind] for ind in np.where(inds)[0]]
        
        rat['trials'][day] = {k: rat['trials'][day][k][inds, ...] for k in rat['trials'][day].keys() if (k not in ['times', 'kinematics', 'peaktimes', 'valtimes'])}
        rat['trials'][day]['kinematics'] = kinematics
        rat['trials'][day]['times'] = times
        
        if 'wds' in rat['name']:
            rat['trials'][day]['peaktimes'], rat['trials'][day]['valtimes'] = peaktimes, valtimes
    
        if sum(inds) < mintrial:
            todel.append(day)
    for day in todel:
        del rat['trials'][day]
    
    ### subselect neural activity ###
    for unum, u in rat['units'].items(): #for each unit
        days = list(u.keys())
        
        for day in days:
            if day in todel:
                del rat['units'][unum][day]
            
            else:
                inds = all_inds[day]
                if sum(rat['units'][unum][day]['keep'][inds]) >= mintrial:
                    rat['units'][unum][day]['raster'] = [rat['units'][unum][day]['raster'][ind] for ind in inds]
                    rat['units'][unum][day]['keep'] = rat['units'][unum][day]['keep'][inds]
                    assert len(rat['units'][unum][day]['raster']) == len(rat['trials'][day]['ipis'])
                else:
                    del rat['units'][unum][day]
    return rat


def keep_warped(rat):
    for unum, u in rat['units'].items(): #for each unit
        for day in u.keys():
            del rat['units'][unum][day]['raster']
            del rat['units'][unum][day]['all']
            
    for day in rat['trials'].keys():
        del rat['trials'][day]['kinematics']
        
    return rat

#%%

if __name__ == '__main__':
    t0 = time.time()
    
    if len(sys.argv) > 1:
        ratname = sys.argv[1]
    else:
        ratname = 'Hindol'

    trim = False
    if len(sys.argv) > 2:
        if sys.argv[2] == 'trim':
            trim = True
            print("trimming")
    
    rat = pickle.load(open('./data/'+ratname+'_data.p', 'rb'))
    rat = parse(rat, trim = trim)#, subselect = False)

    rat = keep_warped(rat)
    if trim:
        pickle.dump(rat, open('./data/'+ratname+'_data_trimmed.p', 'wb'))
    else:
        pickle.dump(rat, open('./data/'+ratname+'_data_warped.p', 'wb'))
    
    print(time.time()-t0)