#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 08:29:08 2021

@author: kris
"""

import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
from plot_utils import panel_font, col_un, col_stab
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cm = 1/2.54

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rcParams['font.sans-serif'] = ['arial']

#%%  Plot variation of Clopath 2017 (left-hand side of panel b)  ######

t = [0, 1, 3, 5, 7]
x1 = [1, 0.8, 0.80, 0.80, 0.80]  # attractor
x2 = [1, 0.8, 0.60, 0.6*0.6/0.8, 0.6*(0.6/0.8)**2]  # drift
x3 = [1, 0.8, 0.80, 0.60, 0.60]  # multi-attractor

t = np.arange(5)
x1 = [0.9 for i in range(5)]  # attractor
x2 = [0.9*(0.7/0.9)**i for i in range(5)]  # drift


plt.figure(figsize=(5.25*cm, 4.8*cm))
plt.plot(t, x2, color = col_un, ls = '-', alpha=0.8)
plt.plot(t, x1, color = col_stab, ls = '-', alpha=0.8)
plt.xlabel('time difference')
plt.ylabel('similarity of\nneural activity', labelpad=-10)
plt.ylim(0.0, 1)
plt.yticks([0, 1], [0, 1])
plt.xticks(t, t)
plt.xticks([])
plt.xlim(t[0], t[-1])
plt.legend(['unstable', 'stable'], frameon = False, loc = 'lower left', bbox_to_anchor=(-0.03, -0.03),
           labelspacing = 0.3)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig('../paper_figs/fig_intro/intro_model_comparison.pdf',
            dpi=240, bbox_inches='tight')
plt.show()

# %% plot behavioral similarity

X = np.arange(-1.5, 1.5, 0.05)
Y = np.arange(-1.5, 1.5, 0.02)
X, Y = np.meshgrid(X, Y)
Z = 0.5*np.exp(-(Y+0.7)**2/0.6)+0.6*np.exp(-(Y-0.7)**2/0.3)  # attractor


fig = plt.figure(figsize=(3.8, 3.8))
ax = fig.gca(projection='3d')

for i in range(0, X.shape[1], 3):
    col = 0.65*np.ones(3)*i/X.shape[1]+0.00
    ax.plot3D(X[:, i], Y[:, i], Z[:, i], c = col)

ax.view_init(elev=70, azim=240)
plt.axis('off')

plt.savefig('../paper_figs/fig_intro/intro_behav_dist.pdf',
            dpi=240, bbox_inches='tight')
plt.show()


# %% plot simpler example behaviors

labels = ['paw_L_x', 'paw_R_x', 'paw_L_y', 'paw_R_y']
name = 'Dhanashri'
rat = pickle.load(open('../data/'+name+'_data_warped.p', 'rb'))
ds = [2, 16, 30]
cs = [0.0, 0.2, 0.5]

plt.figure(figsize=(6.75*cm, 4.0*cm))
d0 = np.amin(list(rat['trials'].keys()))
ds = [d+d0 for d in ds]
inds = np.arange(48, 157)
for k, d in enumerate(ds):
    newtraj = rat['trials'][d]['kinematics_w']['paw_L'][..., 0]
    newtraj = np.mean(newtraj, axis = 0)[inds]
    c = np.ones(3)*cs[k]
    ts = np.linspace(-0.1, 0.8, len(newtraj))

    plt.plot(ts, newtraj, ls = '-', color = c)
plt.axvline(0.0, ls = ':', color = 'k')
plt.axvline(0.7, ls = ':', color = 'k')
plt.xticks([0, 0.7], ['press 1', 'press 2'])
plt.yticks([])
plt.ylabel('position (a.u.)')
plt.legend(['day 0', 'day 14', 'day 28'], ncol = 3, frameon = False,
           loc = 'upper center', bbox_to_anchor = (0.465, 1.225), fontsize = 10,
           columnspacing = 0.8, handletextpad = 0.4, handlelength = 1)
plt.savefig('../paper_figs/fig_intro/behav_example.pdf',
            dpi=240, bbox_inches='tight')
plt.show()
plt.close()


