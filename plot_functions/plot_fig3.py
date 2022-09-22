#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:12:54 2021

@author: kris
"""
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi, global_cmap
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

fig = plt.figure(figsize = (16*cm, 10*cm))


### plot example kinematics ###
gs = fig.add_gridspec(1, 4, left=0, right=0.46, bottom=0.65, top=1., wspace = 0.1, hspace = 0.20, width_ratios = [1, 0.05, 1, 0.07])

labels = [('paw_L', 0), ('paw_L', 1)]
name = 'Dhanashri'
kin_type = 'kinematics_w'
rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))
d0 = 0
d1 = d0+44

days = np.sort(list(rat['trials'].keys()))[d0:d1]
inds = np.arange(48, 157)
for ilab, label in enumerate(labels):
    trajecs = [rat['trials'][day][kin_type][label[0]][..., label[1]] for day in days]
    warps = [rat['trials'][day]['ipis']/0.7 for day in days]
    newdays = [days[j] for j in range(0, len(days)) if len(trajecs[j]) > 0]
    trajec = [np.array(traj)[:, inds] for traj in trajecs if len(traj) > 0]
    plottrajec = np.concatenate(trajec)
    warps = np.concatenate(warps)
    
    vmin, vmax = np.nanquantile(plottrajec, 0.01), np.nanquantile(plottrajec, 0.99)
    plottrajec -= vmin
    plottrajec *= 100/(vmax-vmin)

    ax = fig.add_subplot(gs[0, 2*ilab])
    im = ax.imshow(plottrajec, cmap=global_cmap, aspect='auto', vmin=0, vmax=100, interpolation = 'none')
    ax.set_xticks(np.linspace(plottrajec.shape[1]/9, plottrajec.shape[1], 5)-0.5)
    ax.set_xticklabels(np.round(np.linspace(0.0, 0.8, 5), 1).astype(str))
    ax.set_yticks([0, len(plottrajec)])
    ax.set_yticklabels([1, newdays[-1]-newdays[0]+1])
    ax.set_xlabel('time (s)', labelpad=5)
    ax.set_ylabel('day (expert rat)', labelpad=-15)
    
    N = len(plottrajec)
    xval = plottrajec.shape[1]/9 - 0.5
    ax.annotate("", xy=(xval, 0.025*N), xytext=(xval, -0.075*N), arrowprops=dict(arrowstyle="-|>", fc = 'k'))
    xval = plottrajec.shape[1]/9*8 - 0.5
    ax.annotate("", xy=(xval, 0.025*N), xytext=(xval, -0.075*N), arrowprops=dict(arrowstyle="-|>", fc = 'k'))
    
    if label[1] == 1:
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.set_title('vertical', fontsize = plt.rcParams['font.size'])
    else:
        ax.set_title('horizontal', fontsize = plt.rcParams['font.size'])

newax = fig.add_subplot(gs[0, 3])
newax.set_xticks([])
newax.set_yticks([])
cbar = plt.colorbar(im, orientation = 'vertical', ax = newax, fraction = 1.0, shrink = 1, ticks=[0, 100])
cbar.ax.set_yticklabels(['min', 'max'], rotation = 0, va = 'center', ha = 'left')

### plot days of recording ###
gs = fig.add_gridspec(1, 2, left=0.6, right=1.0, bottom=0.6, top=1., wspace = 0.45, hspace = 0.20)

unit_rats = ['Hindol', 'Hamir']  
titles = ['DLS', 'motor cortex']  
all_durations = []
for i, name in enumerate(unit_rats):
    rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))
    
    ax = fig.add_subplot(gs[0, i])
    y = 0
    
    unums = np.array(list(rat['units'].keys()))
    days = [np.sort(list(rat['units'][unum].keys())) for unum in unums]
    day0s = np.array([day[0] for day in days if len(day) >= 1])
    dayns = np.array([day[-1] for day in days if len(day) >= 1])
    day0s, dayns = day0s - np.amin(day0s), dayns - np.amin(day0s)
    durations = dayns - day0s
    arg1 = np.argsort(durations)
    day0s, dayns = day0s[arg1], dayns[arg1]
    arg2 = np.argsort(day0s)
    day0s, dayns = day0s[arg2], dayns[arg2]
    
    for u in range(len(day0s)):
        if dayns[u] - day0s[u] >= 1:
            ax.plot([day0s[u], dayns[u]],[y, y], 'k-', linewidth=0.5)
            y += 1
    
    ax.set_ylim([-1, y])
    ax.set_xlim(0, np.amax(dayns))

    ax.set_yticks([0, y-1])
    ax.set_yticklabels([1, y])
    ax.set_xticks([0, np.amax(dayns)])
    
    ax.set_xlabel('time (d.)', labelpad = -10)
    if i == 0:
        ax.set_ylabel('unit', labelpad = -15)

    ax.set_title(titles[i], fontsize = plt.rcParams['font.size'])

### plot mean trajectory for all rats ###
gs = fig.add_gridspec(2, 6, left=-0.02, right=0.63, bottom=0.0, top=0.4, wspace = 0.00, hspace = 0.00)

names = ['Hamir', 'Gorakh', 'Gandhar', 'Hindol', 'Dhanashri', 'Jaunpuri']
axs = []
inds = np.arange(48, 157)
durations = []
all_warps = []
for iname, name in enumerate(names):
    rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))
    warps = [rat['trials'][day]['ipis']/0.7 for day in rat['trials'].keys()]
    all_warps.append(np.concatenate(warps))
    for ihand, hand in enumerate(['paw_L', 'paw_R']):
        ax = fig.add_subplot(gs[ihand, iname])
        axs.append(ax)
        
        trajecs = [rat['trials'][day]['kinematics_w'][hand][:, inds, :] for day in rat['trials'].keys()]

        trajecs = np.mean(np.concatenate(trajecs, axis = 0), axis = 0) #mean trajectory
        trajecs = (trajecs - np.amin(trajecs, axis = 0))
        trajecs /= np.amax(trajecs, axis = 0, keepdims = True)

        ax.plot(trajecs[:, 0], trajecs[:, 1], 'k-', alpha = 1/1)#(i_w+1))
        
        taps = np.array([12, 96])
        cs = [[0, 1/4, 3/4], [0, 2/4, 2/4], [0, 3/4, 1/4]]
        cs = [[0,0,0], [0.4,0.4,0.4], [0.7,0.7,0.7]]
        ax.plot(trajecs[0, 0], trajecs[0, 1], 'o', color = cs[0], markersize = 6)
        for itap, tap in enumerate(taps):
            ax.plot(trajecs[taps[itap], 0], trajecs[taps[itap], 1], 'x', color = cs[itap+1],
                    markersize = 7, mew = 1.5)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        if iname == 0 and ihand == 1:
            ax.set_xlabel('horizontal')
            ax.set_ylabel('vertical')
        else:
            ax.axis('off')
        
        buf = 0.15
        ax.set_xlim(-buf, 1+buf)
        ax.set_ylim(-buf, 1+buf)
        
        if ihand == 0:
            ax.plot([0, 1], [1.145, 1.145], 'k-', lw = 1)
            if iname in [0, 5]:
                ax.text(0.5, 1.00, 'animal '+str(iname+1), transform=ax.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize = 11)
        
    ### also store recording durations which we will need in a bit ###
    days = [np.sort(list(rat['units'][unum].keys())) for unum in rat['units'].keys()]
    durations.append(np.array([day[-1]-day[0]+1 for day in days if len(day) > 0]))

all_warps = np.concatenate(all_warps)
print('warping coefficients:', np.mean(all_warps), np.std(all_warps))

### plot histogram of durations ###
gs = fig.add_gridspec(1, 2, left=0.75, right=1.01, bottom=0.1, top=0.4, wspace = 0.2, hspace = 0.00)
    
labs = ['MC', 'DLS']
titles= ['motor cortex', 'DLS']
inds = [range(3), range(3,6)]
bins = np.arange(0, 45, 5)
bins = np.linspace(0, 45, 6)
bins = np.linspace(0, 40, 5)
xvals = (bins[1:] + bins[:-1])/2

bins = [0, 2, 4, 8, 16, 32, 64]
for i in range(2):
    ls = np.concatenate([durations[ind] for ind in inds[i]])

    bins = [0, 3, 9, 27, int(np.amax(ls))]
    xvals = np.arange(len(bins)-1)
    counts, _ = np.histogram(ls, bins = np.array(bins)+1e-2)

    ax = fig.add_subplot(gs[0, i])
    ax.bar(xvals, counts, color = 'k', width = 0.8) # (bins[1]-bins[0])*0.8)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 1000)
    
    if i == 0:
        ax.set_yticks([1, 10, 100, 1000])
        ax.set_yticklabels(['1', '10', '100', '1000'])
        ax.set_ylabel('frequency', labelpad = -5)
    else:
        ax.set_yticks([])
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    
    ax.set_xticks(np.arange(len(bins))-0.5)
    ax.set_xticklabels([str(b) for b in bins], fontsize = 10)
    ax.set_xlabel('duration (d.)', fontsize = 10)
    ax.set_title(titles[i], fontsize = plt.rcParams['font.size'], pad = 9)

    rects = ax.patches
    for i, rect in enumerate(rects):
        ax.text(
            rect.get_x() + rect.get_width() / 2, counts[i]*1.1, str(counts[i]), ha="center", va="bottom", fontsize=8
        )
    
### print number of task-modulated neurons ###

for wds in [False, True]:
    for name in ['DLS', 'MC']:
        if wds:
            name += '_wds'
            same_sims = pickle.load(open('../results/sameday_sim/sims_wds_split.p', 'rb'))
        else:
            same_sims = pickle.load(open('../results/sameday_sim/sims_twotap_split.p', 'rb'))

        unums = np.sort(list(same_sims[name].keys()))
        sig_mod = []
        for unum in unums:
            if same_sims[name][unum] >= 0.15:
                sig_mod.append(unum)
        print(name, ' significantly modulated:', len(sig_mod), 'of', len(unums))

### 
plt.text(-0.05, 1.07, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.05, 0.47, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.53, 1.07, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.66, 0.47, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
    
plt.savefig('../paper_figs/main_fig3.pdf', bbox_inches = 'tight')
plt.close()



