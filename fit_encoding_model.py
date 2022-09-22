
#%% import and load some stuff

import numpy as np
from scipy.stats import pearsonr, linregress, binned_statistic, sem, poisson, ttest_1samp
from scipy.optimize import minimize
import pickle
from utils import load_rat, fit_asymptotic_model, asymptotic_model, calc_exp_ind, fit_model, global_params
from neural_behav_glm import gen_synthetic_rat, sample_data

#%%

names = ['DLS', 'MC']
ms, ss = [], []
for name in names:
    rat = load_rat(name)
    syn_rat = gen_synthetic_rat(rat)

    ### consider running a permutation test as well!!!
    dec_unums, dec_dts, dec_perfs, alphas, cor0s = [], [], [], [], []
    alpha_us = []
    for unum, unit in syn_rat['units'].items():
        days = np.sort(list(unit.keys()))
        dts = []
        perfs = []
        if len(days) > 0:
            peth_0 = unit[days[0]]['peth_w_t']
            mean_act = np.mean(peth_0)
        cor0 = np.nan
        for iday, day in enumerate(days):
            if unit[day]['glm']:
                rate = unit[day]['glm_rate_w'].flatten()
                peths = unit[day]['peth_w_t'].flatten()
                #compute quality of fit
                if sum(peths) >= global_params['minspike']:
                    sim = pearsonr(np.sqrt(peths), np.sqrt(rate))[0]
                    perfs.append(sim)
                    dts.append(day - days[0])
                    if iday == 0:
                        cor0 = sim
        if len(perfs) > 0:
            dec_unums.append(unum)
            dec_perfs.append(perfs)
            dec_dts.append(dts)
            cor0s.append(cor0)
            if len(dts) >= global_params['min_datapoints']+1:
                s, i, _, _, _ = linregress(dts[1:], perfs[1:])
                alpha = s #simply store slope
                if not np.isnan(alpha):
                    alphas.append(alpha)
                alpha_us.append(unum)

    cor0s = np.array(cor0s)
    data = {'alphas': alphas,
            'unums': dec_unums,
            'dts': dec_dts,
            'perfs': dec_perfs,
            'cor0s': cor0s,
            'alpha_us': alpha_us}
    pickle.dump(data, open('./results/decoding_and_cca/glm_decoding_'+name+'.p', 'wb'))

