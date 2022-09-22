import numpy as np
from utils import load_rat, global_params
from scipy.stats import pearsonr
from parse_rat import calc_peth
import time
import pickle

def calc_sameday_sim(split = True):
    np.random.seed(10301002)

    names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gorakh', 'Gandhar', 'DLS', 'MC']
    niter = 1 if split else 100

    for wds in [False, True]: #for each task

        if wds:
            tmin, tmax = -0.2, 0.5
        else:
            tmin, tmax = -0.1, 0.8

        data = {} #store the data
        for iname, name in enumerate(names): #for each animal
            if wds: name += '_wds'
            rat = load_rat(name)
            data[name] = {}

            count = 0
            for unum, unit in rat['units'].items(): #for each unit
                count += 1
                days = np.sort(list(unit.keys())) #consider all days

                if len(days) > 0: #make sure we have data
                    u_sims = np.zeros(niter) #avg similarity for each repeat

                    for iter in range(niter):
                        newsims = [] #similarity for each day
                        for iday, day in enumerate(days): #consider all days

                            raster_w = unit[day]['raster_w'] #get corresponding raster
                            ntrial = len(raster_w) #number of trials
                            if split: #alternating trials
                                inds1 = np.arange(0, ntrial, 2)
                                inds2 = np.arange(1, ntrial, 2)
                            else:
                                #resample twice
                                inds1 = np.random.choice(range(ntrial), ntrial, replace = True) #resample
                                inds2 = np.random.choice(range(ntrial), ntrial, replace = True) #resample

                            # collect rasters
                            rasters = [[raster_w[ind] for ind in inds] for inds in [inds1, inds2]]
                            # compute peths
                            peths = [calc_peth(np.concatenate(raster), tmin, tmax, wds = wds) for raster in rasters]
                            #compute similarity if we have enough spikes
                            if min(np.sum(peths[0]), np.sum(peths[1])) >= global_params['minspike']:
                                newsims.append(pearsonr(peths[0], peths[1])[0])

                        u_sims[iter] = np.mean(newsims) #average across days
                    data[name][unum] = u_sims #average across resamples

        splitstr = '_split' if split else ''
        if wds:
            pickle.dump(data, open('./results/sameday_sim/sims_wds'+splitstr+'.p', 'wb'))
        else:
            pickle.dump(data, open('./results/sameday_sim/sims_twotap'+splitstr+'.p', 'wb'))                

if __name__ == '__main__':
    calc_sameday_sim(split = True)
