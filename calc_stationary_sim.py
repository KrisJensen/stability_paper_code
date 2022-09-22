import copy
import numpy as np
from utils import global_params, load_rat
from scipy.stats import pearsonr
from parse_rat import calc_peth

#also resample all individual sessions with replacement and compute the self-peth
def resample_sessions(rat, nsample = 1000):
    #for all sessions with >= minspikes spikes
    #resample trials with replacement and compute the mean correlation of self-peths
    #average across all sessions for each unit
    us = np.sort(list(rat['units'].keys()))
    all_sims = {}
    mean_sims = {}
    for u in us:
        all_sims[u] = {}
        mean_sims[u] = []
        udict = rat['units'][u]
        for day in udict.keys():
            ntrials = len(udict[day]['raster_w'])
            peth_w_t = udict[day]['peth_w_t'] #peth per trial
            if np.sum(peth_w_t) >= global_params['minspike']:
                all_sims[u][day] = []
                for n in range(nsample):
                    inds1, inds2 = [np.random.choice(np.arange(ntrials), ntrials, replace = True) for _ in range(2)]
                    r1, r2 = [peth_w_t[inds, :] for inds in [inds1, inds2]]
                    p1, p2 = [np.sum(r, axis = 0) for r in [r1, r2]]
                    corr = pearsonr(p1, p2)[0]
                    corr = 0 if np.isnan(corr) else corr
                    all_sims[u][day].append(corr)
                mean_sims[u].append(np.nanmean(all_sims[u][day]))
    return all_sims, mean_sims

def resample_rat(rat):
    '''simply resample all sessions with replacement
    only consider neural data'''
    if 'wds' in rat['name']:
        wds = True
        tmin, tmax = -0.2, 0.5
    else:
        wds = False
        tmin, tmax = -0.1, 0.8
    newrat = {'name': rat['name']}
    newrat['units'] = {} #neural data

    for unum, unit in rat['units'].items():
        newrat['units'][unum] = {}

        days = np.sort(list(unit.keys()))
        if len(days) > 0:
            trial_nums = [len(unit[day]['raster_w']) for day in days]
            ntrial = np.sum(trial_nums)

            big_rast = [unit[day]['raster_w'] for day in days]
            big_rast = [item for sublist in big_rast for item in sublist]
            assert ntrial == len(big_rast)
            for iday, day in enumerate(days):
                inds = np.random.choice(range(ntrial), trial_nums[iday], replace = True) #resample

                ######
                raster_w = [big_rast[ind] for ind in inds] #new raster
                peth_w_t = np.array([calc_peth(raster, tmin, tmax, wds = wds, gauss = False) for raster in raster_w])
                raster_w = np.concatenate(raster_w)
                peth_w = calc_peth(raster_w, tmin, tmax, wds = wds)
                ######

                newrat['units'][unum][day] = {'peth_w_t': peth_w_t, 'peth_w': peth_w}
    return newrat

