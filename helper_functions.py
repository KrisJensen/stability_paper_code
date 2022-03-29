#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:51:50 2019

@author: kris
"""

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import pickle
    
def get_modes(ratname, predominant = False):
    
    if 'wds' in ratname: return [1]
    
    if ratname == 'JaunpuriL': ratname = 'Jaunpuri'
    
    if predominant:
        a = {
                'Hindol':[4],
                'Dhanashri':[2],
                'Jaunpuri':[2],
                'SW166':[3],
                'SW160':[3],
                'SW116':[1],
                'Hamir':[2],
                'Gandhar':[1],
                'Gorakh':[1]
                
                }
        
    else:
        a = {
                'Hindol':range(1,9),
                'Dhanashri':range(1,4),
                'Jaunpuri':range(1,8),
                'SW166':range(1,5),
                'SW160':range(1,5),
                'SW116':range(1,5),
                'Hamir':range(1,5),
                'Gandhar':range(1,3),
                'Gorakh':range(1,5)
                
                }
    
    
    return a[ratname]

    
    