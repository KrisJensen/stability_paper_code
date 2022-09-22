import subprocess as sb
import os
import time

t0 = time.time() #total runtime ~24 hours
pystr = 'python'
pystr = '/scratches/enigma/ktj21/anaconda3/bin/python'

### parse data for all animals ###
print('\n\nparsing data')
sb.call([pystr, 'parse_all.py'])
sb.call([pystr, 'parse_all.py', 'trim'])

### run all RNN analyses ###
print('\n\nrunning RNN analyses')
sb.call([pystr, 'run_all_rnn.py'])

### run all experimental analyses ###
print('\n\nrunning experimental analyses')
sb.call([pystr, 'run_all_exp.py'])

### plot all figures ###
print('\n\nplotting all figures')
os.chdir('plot_functions/')
sb.call([pystr, './plot_all.py'])

print('total runtime:', (time.time()-t0)/60./60., 'hours')
