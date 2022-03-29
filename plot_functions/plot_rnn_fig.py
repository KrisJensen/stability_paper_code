#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:46:26 2021

@author: kris
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import numpy as np
from scipy.stats import pearsonr, binned_statistic, ttest_ind
from scipy.ndimage import gaussian_filter1d
#from PIL import Image

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 11}
plt.rc('font', **font)
from plot_utils import panel_font, png_dpi
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

fig = plt.figure(figsize = (16*cm, 9*cm))
sub = 1


###tf 2.7
nexs = [19, 6, 11]
#nexs = [17, 14, 18]
nexs = [10, 18, 11]

nexs = [15,17,18]
#nexs = [15,10,11]

### embed pdf schematic ###
#gs = fig.add_gridspec(1, 1, left=0, right=0.20, bottom=0.62, top=1., wspace = 0.25, hspace = 0.20)


### plot behavioral responses ###


data_iid = pickle.load(open('../results/rnn/iid_analyses.p', 'rb'))


gs = fig.add_gridspec(1, 1, left=0.00, right=0.28, bottom=0.72, top=1., wspace = 0.25, hspace = 0.20)
y  = data_iid['y']
inds = np.arange(12, 238)
vel = np.mean(y[..., inds], axis = 0)
std = 2*np.std(y[..., inds], axis = 0)
ts = np.arange(len(inds)) / (len(inds)-1)

ax = fig.add_subplot(gs[0, 0])
#ax.text(-0.20, 1.20, 'A', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
for i in range(3):
    ax.plot(ts, vel[i, :], lw = 1, color = 'k')
    ax.fill_between(ts, vel[i, :] - std[i, :], vel[i, :]+std[i, :], alpha = 0.2, color = 'k')
ax.set_xlabel('trial time (a.u.)')
ax.set_ylabel('output (a.u.)')
ax.set_yticks([])


### plot 3 example rasters and/or PETHs ####
gs = fig.add_gridspec(3, 2, left=0.34, right=0.73, bottom=0.55, top=1., wspace = 0.25, hspace = 0.20)
ex_spikes, ex_peths = data_iid['ex_spikes'], data_iid['ex_peth']
tmax = ex_spikes.shape[-1]-1

cols = ['k', np.array([199, 109, 0])/256, np.array([146, 0, 214])/256]
for i_n, n in enumerate(nexs):
    ax = fig.add_subplot(gs[i_n, 0])
    for t in range(0, 100, sub):
        rast = np.nonzero(ex_spikes[t, n, :])[0][::sub]
        ax.scatter(rast, -np.ones(len(rast))*t, color = cols[i_n], s = 2, alpha = 1,  marker='o', linewidths=0.)
    if i_n == 0:
        #ax.text(-0.20, 1.30, 'A', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
        ax.set_title('simulated units', fontsize = plt.rcParams['font.size'])
    elif i_n == 2:
        ax.set_xlabel('trial time')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('trial', labelpad = -0.5)
    ax.set_xlim(0, tmax)
    
ax = fig.add_subplot(gs[:, 1])
ts = np.arange(ex_peths.shape[-1])
for i_n, n in enumerate(nexs):
    ax.plot(ts, ex_peths[n, :], color = cols[i_n], ls = '-')
ax.set_title('simulated PETHs', fontsize = plt.rcParams['font.size'])
ax.set_xticks([])
ax.set_xlabel('trial time')
ax.set_ylabel('activity', labelpad = -0.5)
ax.set_yticks([])
ax.set_xlim(0, tmax)

### plot full heatmap of rasters
gs = fig.add_gridspec(1, 1, left=0.81, right=1.00, bottom=0.55, top=1., wspace = 0.25, hspace = 0.20)

print(ex_spikes.shape)
plot_peths = np.array([ex_spikes[0::2, ...], ex_spikes[1::2, ...]])
plot_peths = np.sum(plot_peths, axis = 1) #across trials; 2xNxt
plot_peths = plot_peths[:, np.sum(plot_peths, axis = (0,-1)) >= 100, :]

ex_zs = np.array([np.mean(plot_peths[..., 5*i:5*(i+1)], axis = -1) for i in range(45)]) #45x2xN
ex_zs = np.transpose(ex_zs, (1,2,0)) #2xNx45
print(ex_zs.shape)
ex_zs = (ex_zs - np.mean(ex_zs, axis = -1)[..., None]) / np.std(ex_zs, axis = -1)[..., None]
plot_zs = ex_zs[0, np.argsort(np.argmax(ex_zs[1, ...], axis = 1)), :] #sort according to held-out trials

ax = fig.add_subplot(gs[0, 0])
ax.imshow(plot_zs, cmap = 'coolwarm', aspect = 'auto', interpolation = 'none', vmin = -2.5, vmax = 2.5)
ax.set_xlabel('trial time')
print('setting yticks!',[1, plot_zs.shape[0]] )
ax.set_xticks([])
ax.set_yticks([1, plot_zs.shape[0]])
ax.set_yticklabels([plot_zs.shape[0], 1])
ax.set_ylabel('simulated unit', labelpad = -15)

### plot correlation and CCA similarity for 2 iid networks ####
gs = fig.add_gridspec(2, 1, left=0.00, right=0.17, bottom=-0.00, top=0.46, wspace = 0.25, hspace = 0.65)

r_same, r_diff = data_iid['r_same'], data_iid['r_diff']
cc_same, cc_diff = [data_iid['CCs'][:, i] for i in range(2)]
#print(cc_diff)
sig = 10

bins = np.linspace(-1, 1, 501)
rs = (bins[1:] + bins[:-1])/2

types = ['single unit', 'latent']
for idat, data in enumerate([[r_same, r_diff], [cc_same, cc_diff]]):

    h_same, h_diff = [np.histogram(dat, bins = bins)[0].astype(float) for dat in data]
    c_same, c_diff = [gaussian_filter1d(h, sig, mode = 'nearest') for h in [h_same, h_diff]]
    c_same, c_diff = [c/np.sum(c)/(rs[1]-rs[0]) for c in [c_same, c_diff]] #normalize

    #ax = fig.add_subplot(gs[0, idat])
    ax = fig.add_subplot(gs[idat, 0])
    ax.plot(rs, c_same, 'c-')
    ax.plot(rs, c_diff, 'm-')
    ax.axvline(np.sum(rs*c_same/sum(c_same)), color = 'c', ls = '-')
    ax.axvline(np.sum(rs*c_diff/sum(c_diff)), color = 'm', ls = '-')
    ax.set_xlim(-1,1)
    maxval = np.amax(np.concatenate([c_same, c_diff]))
    ax.set_ylim(-maxval*0.02, maxval*1.1)
    ax.set_yticks([])
    
    fsize = plt.rcParams['font.size']
    if idat == 0:
        #ax.text(-0.20, 1.25, 'B', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
        #ax.set_title('single unit', fontsize = plt.rcParams['font.size'])
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(['-1', '1'], fontsize = fsize)
        ax.set_xlabel('single unit correlation', labelpad = -2, fontsize = fsize)
        ax.set_ylabel('frequency', fontsize = fsize)
        ax.legend(['same RNN', 'independent'], frameon = False, ncol = 1, loc = 'upper left', bbox_to_anchor = (0.0, 1.75), fontsize = 8)
    else:
        #ax.set_title('latent', fontsize = plt.rcParams['font.size'])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'], fontsize = fsize)
        ax.set_xlim(0, 1)
        ax.set_xlabel('latent similarity', fontsize = fsize, labelpad = -2)
        ax.set_ylabel('frequency', fontsize = fsize)
        
    print(types[idat]+' similarity:', ttest_ind(data[0], data[1], 0))
    print(types[idat]+' n = ('+str(len(data[0]))+', '+str(len(data[1]))+')')


### plot interpolation schematic ###
# gs = fig.add_gridspec(1, 1, left=0.53, right=1.0, bottom=0.32, top=0.65, wspace = 0.15, hspace = 0.20)

# im = Image.open('../results/rnn/interpolate-1.png')
# height = im.size[1]
# # We need a float array between 0-1, rather than
# # a uint8 array between 0-255
# im = np.array(im).astype(np.float) / 255
# # With newer (1.0) versions of matplotlib, you can 
# # use the "zorder" kwarg to make the image overlay
# # the plot, rather than hide behind it... (e.g. zorder=10)
# ax = fig.add_subplot(gs[0, 0])
# im = np.flip(im.transpose(1,0,2), axis = 1)
# im = im[250:-300, ...]
# ax.imshow(im)
# ax.text(-0.40, 1.20, 'c', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
# ax.axis('off')


### plot example rasters (2 stable 2 drifting)###
data_s, data_d = [pickle.load(open('../results/rnn/interp_analyses_'+driftstr+'.p', 'rb')) for driftstr in ['stable', 'drift']]

gs = fig.add_gridspec(1, 5, left=0.25, right=0.55, bottom=-0.02, top=0.36, wspace = 0.15, hspace = 0.20, width_ratios = [1,1,0.2,1,1])

iplot = -2
for idat, data in enumerate([data_s, data_d]):
    ex_spikes = data[0]['example_spikes']
    ninterp = ex_spikes.shape[0]
    tmax = ex_spikes.shape[-1]-1
    iplot += 1
    for i_n, n in enumerate(nexs[:2]):
        iplot += 1
        ax = fig.add_subplot(gs[0, iplot])
        yval = 1
        for iday in range(ninterp):
            #col = [0, 0.1+0.8*iday/ninterp, 0.9-0.8*iday/ninterp]
            frac = iday/(ninterp-1)
            if idat == 0:
                col = frac * np.array([0, 1, 1]) + (1-frac)*np.array([0, 0.5, 0.5])
            else:
                col = frac * np.array([1, 0, 1]) + (1-frac)*np.array([0.5, 0, 0.5])

            for itrial in range(0, 100, sub):
                yval -= 1
                rast = ex_spikes[iday, itrial, n, :]
                rast = np.nonzero(rast)[0][::sub]
                ax.scatter(rast, np.ones(len(rast))*yval, color = col, s = 2, alpha = 1,  marker='o', linewidths=0.)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, tmax)
        #ax.set_xlabel('trial time', fontsize = 8)
        if iplot == 0:
            #ax.text(-0.50, 1.20, 'C', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
            ax.set_ylabel('time (days)', labelpad = -10)
            
            yday = yval/ninterp
            ax.set_yticks([yday/2, yval - yday/2])
            ax.set_yticklabels(['1', '7'])
        ax.set_ylim(yval, 0)
        
        ax.set_title(['unit A', 'unit B'][i_n], fontsize = plt.rcParams['font.size'], pad = -0*5)

shift = 0.2
plt.text(0.12+shift, 0.42, 'stable',ha='center', va='bottom',transform=fig.transFigure)
plt.text(0.2825+shift, 0.42, 'drifting',ha='center', va='bottom',transform=fig.transFigure)
con = ConnectionPatch(xyA=(0.055+shift, 0.415), xyB=(0.185+shift, 0.415), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)
con = ConnectionPatch(xyA=(0.2175+shift, 0.415), xyB=(0.3475+shift, 0.415), arrowstyle="-",
                      coordsA='figure fraction', coordsB='figure fraction',
                      axesA=ax, axesB=ax, lw = 1)
ax.add_artist(con)

### plot correlation of activity/behavioral/latent space ###
gs = fig.add_gridspec(1, 3, left=0.60, right=1.00, bottom=0.05, top=0.35, wspace = 0.15, hspace = 0.20)
data_s, data_d = [pickle.load(open('../results/rnn/interp_analyses_rep_'+driftstr+'.p', 'rb')) for driftstr in ['stable', 'drift']]

labs = ['neural', 'behav', 'CCA']
titles = ['PETH', 'behavior', 'latent']
bins = np.arange(0.5, 7.5, 1)
bin_dts = 0.5*(bins[1:] + bins[:-1])
for itype, lab in enumerate(labs):
    ax = fig.add_subplot(gs[0, itype])
    for idata, data in enumerate([data_s, data_d]):
        sims = [data[key][lab+'_sims'] for key in data.keys()]
        dts = [data[key][lab+'_dts'] for key in data.keys()]
        bin_sims = np.zeros((len(sims), len(sims[0]), len(bin_dts)))*np.nan
        for rep in range(len(sims)): #for each repeat
            for n in range(len(sims[rep])): #for each pair of networks
                dt, sim = dts[rep][n], sims[rep][n]
                #print(len(dt), len(sim))
                if len(dt) >= 1:
                    bin_sims[rep, n, :] = binned_statistic(dt, sim, bins = bins)[0]
        bin_sims = np.nanmean(bin_sims, axis = 1) #average over 'output units' (neurons/output dims/neuron groups)
        m, s = np.nanmean(bin_sims, axis = 0), np.nanstd(bin_sims, axis = 0)
        s = s #/ np.sqrt(np.sum(1-np.isnan(bin_sims), axis = 0))

        label = (['stable', 'drifting'][idata] if itype == 1 else None)
        c = ['c', 'm'][idata]
        ax.plot(bin_dts, m, c+['-', '--'][idata], label = label)
        ax.fill_between(bin_dts, m-s, m+s, color = c, alpha = 0.2)
        
    ax.set_xlabel(r'$\Delta$'+'time (d.)', labelpad = -2)#, fontsize = 10)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1,3,5])
    ax.set_xlim(0, 6)
    ax.set_title(titles[itype], fontsize = plt.rcParams['font.size'])
    if itype == 0:
        ax.set_ylabel('similarity', labelpad = -10)
        ax.set_yticks([0, 1])
        #ax.text(-0.30, 1.25, 'D', transform=ax.transAxes,fontsize=panel_font, fontweight='bold', horizontalalignment='left', verticalalignment='top')
    else:
        ax.set_yticks([])
    if itype == 1:
        ax.legend(frameon = False, handletextpad=0.5, handlelength=1.2, loc = 'lower center')

        
plt.text(-0.03, 1.09, 'A',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.30, 1.09, 'B',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.76, 1.09, 'C',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(-0.03, 0.59, 'D',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.22, 0.47, 'E',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
plt.text(0.57, 0.44, 'F',ha='left', va='top',transform=fig.transFigure, fontweight='bold',fontsize=panel_font)
        

#plt.savefig('../paper_figs/fig_rnn.png', bbox_inches = 'tight') 
plt.savefig('../paper_figs/fig_rnn.png', bbox_inches = 'tight', dpi = png_dpi) 
plt.savefig('../paper_figs/fig_rnn.pdf', bbox_inches = 'tight')
#plt.show()
plt.close()

