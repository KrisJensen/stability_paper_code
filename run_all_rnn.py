import subprocess as sb
import os

### create directories ###
print('\n\ncreating directories for results and figures')
os.makedirs('./plots', exist_ok=True)
os.makedirs('./plots/rnn', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/rnn', exist_ok=True)
os.makedirs('./results/rnn/reps', exist_ok=True)

### train RNNs (~2h) ###
print('\n\ntraining RNNs')
sb.call(['python', 'RNN_train_model.py'])

### run analyses on iid networks (~2 mins) ###
print('\n\ncomparing independently trained networks')
sb.call(['python', 'RNN_analysis_pair.py'])

### run analyses of drifting network (~10 mins) ###
print('\n\nruning analysis of drifting network')
sb.call(['python', 'RNN_analysis_interp.py', '1'])

### run analyses of stable network (~10 mins) ###
print('\n\nruning analysis of stable network')
sb.call(['python', 'RNN_analysis_interp.py', '0'])

