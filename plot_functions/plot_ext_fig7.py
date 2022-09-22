import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic
from plot_utils import get_col
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

fig = plt.figure(figsize = (16*cm, 8*cm))

for itask, task in enumerate(['twotap', 'wds']):
    trimstr = '_trimmed'
    if itask == 0:
        dataname = '../results/neural_similarity_analyses'+trimstr+'_twotap.p'
        base_sim = pickle.load(open('../results/behav_similarity_analyses_twotap.p', 'rb'))
        names = ['DLS', 'MC']
        bot, top = 0.60, 1.0
    else:
        dataname = '../results/neural_similarity_analyses'+trimstr+'_wds.p'
        base_sim = pickle.load(open('../results/behav_similarity_analyses_wds.p', 'rb'))
        names = ['DLS_wds', 'MC_wds']
        bot, top = 0.00, 0.40

    ### plot neural similarity vs time

    gs = fig.add_gridspec(1,2, left=0.0, right=0.26, bottom=bot, top=top, wspace = 0.5, hspace = 0.1)
    data = pickle.load(open(dataname, 'rb'))
    ax = fig.add_subplot(gs[0, :]) #fill the whole thing!
    labels = ['DLS', 'MC']
    for iname, name in enumerate(names):
        dts, rec_times, sims, unums = [data[name][k] for k in ['dts', 'rec_times', 'sims', 'unums']]
        
        maxt = 14 if itask == 0 else 10
        inds = (rec_times >= maxt).nonzero()[0]
        bins = np.arange(0.5, maxt - 0.5 + 0.01, 1)
        long_sims = np.zeros((len(inds), len(bins)-1))
        for iind, ind in enumerate(inds):
            unum = unums[ind]
            dt, sim = dts[ind], sims[ind]
            long_sims[iind, :] = binned_statistic(dt, sim, statistic = 'mean', bins = bins)[0]

        nans = np.isnan(long_sims)
        no_nans = np.array([(not any(nans[i, :])) for i in range(nans.shape[0])])
        long_sims = long_sims[no_nans, :]

        m, s = np.nanmean(long_sims, axis = 0), np.nanstd(long_sims, axis = 0)/np.sqrt(np.sum(1-np.isnan(long_sims), axis = 0))
        xs = (bins[1:] + bins[:-1])/2
        ax.plot(xs, m, color = get_col(name), ls = '-', label = labels[iname])
        ax.fill_between(xs, m-s, m+s, color = get_col(name), alpha = 0.2)

    if itask == 0:
        ax.set_ylim(-0.03, 1)
        ax.set_yticks([0, 1])
        ax.set_xticks([1, 6, 12])
    else:
        ax.set_ylim(-0.03, 0.6)
        ax.set_yticks([0, 0.6])
        ax.set_xticks([1, 4, 8])

    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel('time difference (d.)')
    ax.set_ylabel('correlation', labelpad = -10)
    ax.legend(frameon = False)

    ### plot behav similarity vs time

    gs = fig.add_gridspec(1, 1, left=0.37, right=0.63, bottom=bot, top=top, wspace = 0.15, hspace = 0.20)
    name = 'Dhanashri' if itask == 0 else 'Dhanashri_wds'
    ms = []
    ms_w = []
    bins = np.arange(0.5, 31, 1)
    for name in ['Dhanashri', 'Hindol', 'Jaunpuri', 'Hamir', 'Gorakh', 'Gandhar']:
        if itask == 1: name += '_wds'
        dts, sims, day1s = [data[name][k] for k in ['dts_b', 'sims_b', 'day1s_b']]
        dts_w, sims_w, day1s_w = [base_sim[name][k] for k in ['dts', 'sims', 'day1s']]
        m = binned_statistic(dts, sims, statistic = 'mean', bins = bins)[0]
        s = binned_statistic(dts, sims, statistic = 'std', bins = bins)[0]
        m_w = binned_statistic(dts_w, sims_w, statistic = 'mean', bins = bins)[0]
        ms.append(m)
        ms_w.append(m_w)
    ms = np.array(ms)
    m = np.nanmean(ms, axis = 0)
    s = np.nanstd(ms, axis = 0) / np.sqrt(np.sum(~np.isnan(ms), axis = 0))
    m_w = np.nanmean(ms_w, axis = 0)

    xs = (bins[1:]+bins[:-1]) / 2

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xs, m, 'k-', label = 'trim')
    ax.fill_between(xs, m-s, m+s, color = 'k', alpha = 0.2)
    ax.plot(xs, m_w, ls = '--', color = np.ones(3)*0.3, label = 'warp')
    ax.set_xlabel('time difference (days)')
    ax.set_xticks([1, 5, 10])
    ax.set_xlim(xs[0], 30)
    ax.set_ylim(0.7, 1)
    ax.set_yticks([0.7, 1])
    ax.set_xlim(1, 10)

    if itask == 0:
        ax.set_ylabel('velocity corr.', labelpad = -10)
        ax.legend(frameon = False, ncol = 2, handlelength = 1.5, handletextpad = 0.5, columnspacing = 1.0, loc = 'upper center')
    else:
        ax.set_ylabel('acc corr.', labelpad = -10)

    ### plot neural behavior correlation
    wdsstr = '' if itask == 0 else '_wds'

    res = pickle.load(open('../results/neural_behav_corr'+trimstr+'_DLS'+wdsstr+'.p', 'rb'))
    res_mc = pickle.load(open('../results/neural_behav_corr'+trimstr+'_MC'+wdsstr+'.p', 'rb'))

    all_sims_n, all_sims_b = res['all_sims_n'], res['all_sims_b']
    all_corrs = [pearsonr(all_sims_n[i], all_sims_b[i])[0] for i in range(len(all_sims_n))]

    all_sims_n_mc, all_sims_b_mc = res_mc['all_sims_n'], res_mc['all_sims_b']
    all_corrs_mc = [pearsonr(all_sims_n_mc[i], all_sims_b_mc[i])[0] for i in range(len(all_sims_n_mc))]

    gs = fig.add_gridspec(1, 1, left=0.74, right=1.0, bottom=bot, top=top, wspace = 0.15, hspace = 0.20)

    ax = fig.add_subplot(gs[0, 0])
    h, _, _ = ax.hist(all_corrs, bins = np.linspace(-1, 1, 11), color = get_col('DLS'), alpha = 0.4)
    maxval = np.nanmax(h)*1.05
    ax.axvline(np.mean(all_corrs), color = get_col('DLS'), label = 'DLS')

    h_mc, _, _ = ax.hist(all_corrs_mc, bins = np.linspace(-1, 1, 11), color = 'r', alpha = 0.4)
    ax.axvline(np.mean(all_corrs_mc), color = get_col('MC'), label = 'MC')

    ax.set_ylim(0, maxval)
    ax.set_xlabel('correlation')
    ax.set_ylabel('frequency', labelpad = -10)
    ax.set_yticks([0, 25])
    ax.legend(frameon = False, fontsize = 10, handlelength = 1.2, handletextpad = 0.4, loc = 'upper left', borderpad = 0.2)

    
y1, y2 = 1.07, 0.48
x1, x2, x3 = -0.06, 0.29, 0.67
plt.text(x1, y1, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(x2, y1, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(x3, y1, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(x1, y2, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(x2, y2, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(x3, y2, 'F',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)

#%% save fig
plt.savefig('../paper_figs/ext_data_fig7.jpg', bbox_inches = 'tight', dpi = png_dpi)
plt.close()