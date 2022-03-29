import subprocess as sb
import os
import time

t0 = time.time() #total runtime ~3.5 hours

### parse data for all animals ###
print('\n\nparsing data')
sb.call(['python', 'parse_all.py'])

### run all RNN analyses ###
print('\n\nrunning RNN analyses')
sb.call(['python', 'run_all_rnn.py'])

### run all experimental analyses ###
print('\n\nrunning experimental analyses')
sb.call(['python', 'run_all_exp.py'])

### plot all figures ###
print('\n\nplotting all figures')
os.chdir('plot_functions/')
sb.call(['python', './plot_all.py'])

print('total runtime:', (time.time()-t0)/60./60., 'hours')
