
#%%

import numpy as np
import pickle
from utils import load_rat, global_params
from RNN_utils import run_cca
from scipy.stats import pearsonr, linregress, binned_statistic
from sklearn.linear_model import RidgeCV
from utils import fit_model
np.random.seed(9170802)

rat = load_rat('Hindol') #load the data

unums = np.array(list(rat['units'].keys())) #get unit numbers
#list of days each unit was recorded
days = [np.sort(list(rat['units'][unum].keys())) for unum in unums]
day0s = np.array([day[0] for day in days]) #by time of first recording

early_days = np.arange(np.amin(day0s), np.amin(day0s)+7)+7 #days 7-14

#find all units recorded for this set of days
include = np.where([all([d in day for d in early_days]) for day in days])[0] #recorded for all days
#make sure the neurons actually fire in all days
thresh = 1.0
active = [np.mean([ np.sum(rat['units'][unums[i]][day]['peth_w']) >= global_params['minspike'] for day in early_days if day in days[i]]) >= thresh for i in include ]
include = include[np.array(active)]
us = unums[include] #list of units to consiter

#collect all the PETHs in a big matrix
Y = np.array([[rat['units'][unum][day]['peth_w'] for day in early_days] for unum in us]) #neurons by days by t

#normalize activity
Ynorm = (Y - np.mean(Y, axis = (1,2), keepdims = True)) / np.std(Y, axis = (1,2), keepdims = True)
Ynorm = Y #don't actually use the normalized activity because it didn't help

N = Y.shape[0] #number of neurons
dts = [] #time differences
sims_cca = [] #CCA correlations
sims_rho = [] #simple correlations across neurons
sims_pca = [] #PCA correlations
for i1 in range(len(early_days)): #for each day
    for i2 in range(i1+1, len(early_days)): #for each other day
        dts.append(i2 - i1) #time between days

        #only consider neurons with at least minspike neurons on each day
        #including all neurons for CCA doesn't change things qualitatively
        inds = np.where(np.minimum(np.sum(Y[:, i1, :], axis = 1), np.sum(Y[:, i2, :], axis = 1)) >= global_params['minspike'])[0]

        #get activity for the relevant days and neurons
        Y1, Y2 = Ynorm[inds, i1, :], Ynorm[inds, i2, :]

        #run CCA
        _, _, _, sim_pca, sim_cca = run_cca(Y1, Y2, nPC = 3, nCC = 3, return_aligned = True)

        #run CCA without returning the PCA results
        sim_cca = run_cca(Y1, Y2, nPC = len(inds), nCC = len(inds), return_aligned = False)

        #average PETH correlation across neurons
        sim_rho = [pearsonr(Y1[i, :], Y2[i, :])[0] for i in range(len(inds))]

        #store data
        sims_cca.append(sim_cca)
        sims_pca.append(sim_pca)
        sims_rho.append(np.mean(sim_rho))


#turn into numpy array
sims_cca, sims_rho, sims_pca, dts = [np.array(arr) for arr in [sims_cca, sims_rho, sims_pca, dts]]

u_dts = np.unique(dts) #unique time differences
#compute mean and std for each time difference
m_cca = np.array([np.mean(sims_cca[dts == d])for d in u_dts])
s_cca = np.array([np.std(sims_cca[dts == d])/np.sqrt(np.sum(dts == d)) for d in u_dts])
m_rho = np.array([np.mean(sims_rho[dts == d])for d in u_dts])
s_rho = np.array([np.std(sims_rho[dts == d])/np.sqrt(np.sum(dts == d)) for d in u_dts])


#%% compute alphas (with linear regression for now)
s_c, i_c, _, _, _ = linregress(dts, sims_cca)
s_r, i_r, _, _, _ = linregress(dts, sims_rho)

niter = 10000 #bootstrapped samples
alphas_c, alphas_r = np.zeros(niter), np.zeros(niter)
for n in range(niter):
    inds = np.random.choice(np.arange(len(dts)), len(dts), replace = True) #sample with replacement

    #comute stability indices
    s, i, _, _, _ = linregress(dts[inds], sims_cca[inds])
    alphas_c[n] = s/i
    alphas_c[n] = fit_model(dts[inds], sims_cca[inds]).x[1]
    s, i, _, _, _ = linregress(dts[inds], sims_rho[inds])
    alphas_r[n] = s/i
    alphas_r[n] = fit_model(dts[inds], sims_rho[inds]).x[1]

# dump data
data = {
    'dts': dts,
    'sims_rho': sims_rho,
    'sims_cca': sims_cca,
    'alphas_cca': alphas_c,
    'alphas_rho': alphas_r}
pickle.dump(data, open('./results/decoding_and_cca/latent_similarity.p', 'wb'))


# %% try to do some decoding

#trials x timepoints x inputs for each day
#decoding kinematics is easier than velocity
behavior = [np.concatenate([rat['trials'][day]['kinematics_w'][k] for k in ['paw_L', 'paw_R']], axis = -1) for day in early_days]

#timepoints corresponding to neural activity
tmin, tmax, binsize = -0.1, 0.8, global_params['binsize']
tx_bins = np.arange(tmin, tmax+binsize, binsize)

#timepoints corresponding to behavior
tx = rat['trials'][early_days[0]]['times']

# average behavior for each neural bin
# ntrial x nbins x 4
Xs = [np.array([binned_statistic(tx, x_t.T, statistic = 'mean', bins = tx_bins)[0].T for x_t in X]) for X in behavior]

# neural activity (unit x trial x timepoint for each day)
Ys = [np.array([rat['units'][unum][day]['peth_w_t'] for unum in us]) for day in early_days] 

# sqrt transform and convolce (cf gallego)
from scipy.ndimage import gaussian_filter1d
Ys = [gaussian_filter1d(np.sqrt(Y), 2, axis = 1, mode = 'constant') for Y in Ys]

delay = 5 #5 is optimal (100ms)
if delay > 0: #offset behavior from neural activity
    Xs = [X[:, delay:, :] for X in Xs]
    Ys = [Y[:, :, :-delay] for Y in Ys]

# concatenate trials and timebins
# ntrial*nbin x 4
Xs = [np.array([X[..., i].flatten() for i in range(X.shape[-1])]) for X in Xs]
# ntrial*bin x neuron
Ys = [np.array([Y[i, ...].flatten() for i in range(Y.shape[0])]) for Y in Ys]

alphas = 10**np.linspace(-3, 3, 50) #regularization strengths
# initialize lists of results
rhos, rhos_same, dts, rhos_pca, rhos_cca, rhos_sub = [], [], [], [], [], []

def train_eval_ridgeCV(X0, Y0, X1, Y1, alphas):
    #train a model
    model = RidgeCV(alphas = alphas).fit(Y0.T, X0.T)
    Xp = model.predict(Y1.T).T #predict output
    rho = np.mean([pearsonr(X1[i, :], Xp[i, :])[0] for i in range(4)]) #compare to ground truth
    return rho, model

nPC, nCC = 10, 2
for d0 in range(len(early_days)):
    X0, Y0 = Xs[d0], Ys[d0] #data from first day
    rhos_same.append(train_eval_ridgeCV(X0, Y0, X0, Y0, alphas)[0])

    #for d1 in range(len(early_days)): #note sem is not right/ maybe symmetrize first
    for d1 in range(d0+1, len(early_days)):
        X1, Y1 = Xs[d1], Ys[d1] #data for second day

        #test on second day
        rho1, model = train_eval_ridgeCV(X0, Y0, X1, Y1, alphas)
        rho2, model = train_eval_ridgeCV(X1, Y1, X0, Y0, alphas)
        rhos.append(np.mean([rho1, rho2]))
        dts.append(d1-d0)

        #match the number of timepoints (trials) on each day (note that Gallego et al. also do this)
        nt = min(Y0.shape[1], Y1.shape[1]) #minimum trial number

        #run PCA and CCA
        L0, L1, L1_aligned, sim_pca, sim_cca = run_cca(Y0[:, :nt], Y1[:, :nt], nPC = nPC, nCC = nCC, return_aligned = True)
        rho_pca1, m_pca1 = train_eval_ridgeCV(X0[:, :nt], L0, X1[:, :nt], L1, alphas)
        rho_pca2, m_pca2 = train_eval_ridgeCV(X1[:, :nt], L1, X0[:, :nt], L0, alphas)
        rhos_pca.append(np.mean([rho_pca1, rho_pca2]))

        # run CCA (also align the other way)
        rho_cca1, m_cca1 = train_eval_ridgeCV(X0[:, :nt], L0, X1[:, :nt], L1_aligned, alphas)
        L1, L0, L0_aligned, sim_pca, sim_cca = run_cca(Y1[:, :nt], Y0[:, :nt], nPC = nPC, nCC = nCC, return_aligned = True)
        rho_cca2, m_cca2 = train_eval_ridgeCV(X1[:, :nt], L1, X0[:, :nt], L0_aligned, alphas)
        rhos_cca.append(np.mean([rho_cca1, rho_cca2]))

        #use matched numbers of samples for raw decoding
        rho_sub1, m_sub1 = train_eval_ridgeCV(X0[:, :nt], Y0[:, :nt], X1[:, :nt], Y1[:, :nt], alphas)
        rho_sub2, m_sub2 = train_eval_ridgeCV(X1[:, :nt], Y1[:, :nt], X0[:, :nt], Y0[:, :nt], alphas)
        rhos_sub.append(np.mean([rho_sub1, rho_sub2]))

#turn into np arrays
rhos, rhos_same, dts, rhos_cca = np.array(rhos), np.array(rhos_same), np.array(dts), np.array(rhos_cca)

#compute mean for each dt
u_dts = np.unique(dts)
m_rho = np.array([np.mean([rhos[dts == dt]]) for dt in u_dts])
s_rho = np.array([np.std([rhos[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])

m_cca = np.array([np.mean([rhos_cca[dts == dt]]) for dt in u_dts])
s_cca = np.array([np.std([rhos_cca[dts == dt]])/np.sqrt(sum(dts == dt)) for dt in u_dts])


# %% compute stability indices

niter = 10000
alphas_c, alphas_r = np.zeros(niter), np.zeros(niter)
for n in range(niter):
    inds = np.random.choice(np.arange(len(dts)), len(dts), replace = True) #sample with replacement

    s, i, _, _, _ = linregress(dts[inds], rhos_cca[inds])
    alphas_c[n] = s/i
    alphas_c[n] = fit_model(dts[inds], rhos_cca[inds]).x[1]
    s, i, _, _, _ = linregress(dts[inds], rhos[inds])
    alphas_r[n] = s/i
    alphas_r[n] = fit_model(dts[inds], rhos[inds]).x[1]

data = {
    'dts': dts,
    'rhos_raw': rhos,
    'rhos_cca': rhos_cca,
    'alphas_cca': alphas_c,
    'alphas_raw': alphas_r}
pickle.dump(data, open('./results/decoding_and_cca/population_decoding.p', 'wb'))

