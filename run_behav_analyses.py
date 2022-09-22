import numpy as np
import pickle
import sys
from scipy.stats import pearsonr

from compute_behav_sim import compute_behav_sim
from compute_neural_sim import compute_neural_sim
from compute_neural_behav_sim import comp_neural_behav_sim
from neural_behav_glm import gen_synthetic_rat, sample_data
from utils import load_rat, check_modulated
import os
os.nice(5)

np.random.seed(8191609)

if len(sys.argv) > 1: #0/1 for twotap/wds
    wds = bool(int(sys.argv[1]))
else:
    wds = False
print('wds:', wds)

names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gandhar', 'Gorakh']
data = {}
nreps = 10000

for name in names:
    print(name)
    if wds:
        name += '_wds'
    rat = load_rat(name)
    data[name] = {}
    
    ### compute behavior vs time difference ###
    dts, sims, day1s = compute_behav_sim(rat, verbose = False)
    data[name]['dts'] = dts
    data[name]['sims'] = sims
    data[name]['day1s'] = day1s
    
    ### bootstrap confidence interval ###
    corrs = np.zeros(nreps)
    dts, sims = np.array(dts), np.array(sims)
    for nrep in range(nreps):
        inds = np.random.choice(len(dts), len(dts), replace = True)
        corrs[nrep] = pearsonr(dts[inds], sims[inds])[0]
        
    qs = [0.005, 0.025, 0.25, 0.5, 0.75, 0.975, 0.995]
    data[name]['corrs'] = corrs

if wds:
    pickle.dump(data, open('./results/behav_similarity_analyses_wds.p', 'wb'))
else:
    pickle.dump(data, open('./results/behav_similarity_analyses_twotap.p', 'wb'))
    
names = ['DLS', 'MC']
for name in names:
    ### correlate behavior with neural activity
    
    if wds:
        name += '_wds'
        same_sims = pickle.load(open('./results/sameday_sim/sims_wds_split.p', 'rb'))
    else:
        same_sims = pickle.load(open('./results/sameday_sim/sims_twotap_split.p', 'rb'))
        
    rat = load_rat(name)
    
    dts_b, sims_b, day1s_b = compute_behav_sim(rat)
    unums, dts_n, sims_n, day1s_n = compute_neural_sim(rat, synthetic = False, day0 = False)

    mod_inds, _ = check_modulated(unums, same_sims, name) #only consider significantly modulated
    unums, dts_n, sims_n, day1s_n = [[arr[ind] for ind in mod_inds] for arr in [unums, dts_n, sims_n, day1s_n]]

    all_sims_n, all_sims_b, all_corrs = comp_neural_behav_sim(unums, dts_n, sims_n, day1s_n, dts_b, sims_b, day1s_b)

    pickle.dump({'all_sims_n': all_sims_n, 'all_sims_b': all_sims_b},
                    open('./results/neural_behav_corr_'+name+'.p', 'wb'))
    
    print('mean corr:', np.mean(all_corrs))
    
    ### run permutation test ###
    print('running permutation test')
    means = []
    nreps_nb = 5000
    for i in range(nreps_nb):
        _, _, corrs = comp_neural_behav_sim(unums, dts_n, sims_n, day1s_n, dts_b, sims_b, day1s_b, permute = True)
        means.append(np.mean(corrs))
        if i % 1000 == 0: print('permute', i, np.mean(means), np.std(means))

    ### compare with synthetic data ###
    print('generating synthetic data')
    means_syn = []
    rat = gen_synthetic_rat(rat) #fit GLM
    for i in range(nreps_nb):
        rat = sample_data(rat) #resample neural data
        unums, dts_n, sims_n, day1s_n = compute_neural_sim(rat, synthetic = True, day0 = False)

        mod_inds, _ = check_modulated(unums, same_sims, name) #only consider significantly modulated
        unums, dts_n, sims_n, day1s_n = [[arr[ind] for ind in mod_inds] for arr in [unums, dts_n, sims_n, day1s_n]]

        _, _, corrs = comp_neural_behav_sim(unums, dts_n, sims_n, day1s_n, dts_b, sims_b, day1s_b)
        means_syn.append(np.mean(corrs))
        if i % 1000 == 0: print('synthetic', i, np.mean(means_syn), np.std(means_syn))

    ### save correlation analysis ###
    pickle.dump({'means': means, 'means_syn': means_syn, 'all_corrs': all_corrs},
                open('./results/neural_behav_corr_syn_'+name+'.p', 'wb'))

