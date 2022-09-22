import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
import pickle
from utils import load_rat
from neural_behav_glm import make_design_mat, neg_log_lik_lnp, get_regressors
import copy
import os
os.nice(5)
#%%
np.random.seed(7582109)

names = ['DLS', 'MC']
ms, ss = [], []
for name in names:
    unums = []
    sims = []
    rat = load_rat(name)
    all_unums = np.sort(list(rat['units'].keys()))
    for iu, unum in enumerate(all_unums):
        print('unit:', iu, 'of', len(all_unums))
        unit = rat['units'][unum]
        days = np.sort(list(unit.keys()))
        if len(days) > 3.5:
            d0 = days[0]
            peths, vels, times, keep, type_ = get_regressors(d0, unum, rat)
            inds = np.where(keep)[0]
            pred_rates = []
            true_peths = []
            for ind in inds:
                newbool = copy.copy(keep)
                newbool[ind] = False
                X, Y = make_design_mat(times, peths, vels, type_ = 'twotap', keep = newbool)
                F = X.shape[-1] #number of features
                if np.sum(Y) > F:
                    w0 = np.random.normal(0, .0002, F)
                    # Find parameters that minimize the negative log likelihood function
                    res = minimize(neg_log_lik_lnp, w0, args=(X, Y), options = {'maxiter': 5000, 'maxfun': 100000}, method = 'L-BFGS-B')
                    if res['success']:
                        w = res['x']
                        tinds = np.arange(ind, ind+1) #just the test index
                        X, Y = make_design_mat(times, peths, vels, type_ = 'twotap', keep = tinds)
                        rate = (np.exp(X @ w))
                        pred_rates.append(rate.flatten())
                        true_peths.append(Y.flatten())
            if len(true_peths) > 0.5:
                Y, Yhat = np.concatenate(true_peths).flatten(), np.concatenate(pred_rates).flatten()
                if np.all(np.isfinite(Y)) and np.all(np.isfinite(Yhat)):
                    sim = pearsonr(np.sqrt(Y.flatten()), np.sqrt(Yhat.flatten()))[0]
                    sims.append(sim)
                    unums.append(unum)

    data = {'unums': unums, 'sims': sims}
    pickle.dump(data, open('./results/decoding_and_cca/glm_decoding_crossval_'+name+'.p', 'wb'))
