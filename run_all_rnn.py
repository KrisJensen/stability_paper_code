import subprocess as sb
import os

pystr = 'python'
pystr = '/scratches/enigma/ktj21/anaconda3/bin/python'

### create directories ###
print('\n\ncreating directories for results')
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/rnn', exist_ok=True)
os.makedirs('./results/rnn/reps', exist_ok=True)

### train RNNs (~2h) ###
print('\n\ntraining RNNs')
sb.call([pystr, 'RNN_train_model.py'])

### run analyses on iid networks (~2 mins) ###
print('\n\ncomparing independently trained networks')
sb.call([pystr, 'RNN_analysis_pair.py'])

### run analyses of drifting network (~10 mins) ###
print('\n\nruning analysis of drifting network')
sb.call([pystr, 'RNN_analysis_interp.py', '1'])

### run analyses of stable network (~10 mins) ###
print('\n\nruning analysis of stable network')
sb.call([pystr, 'RNN_analysis_interp.py', '0'])

