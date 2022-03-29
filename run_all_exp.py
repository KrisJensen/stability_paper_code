import subprocess as sb
import os

### create directories ###
print('\n\ncreating directories for results and figures')
os.makedirs('./plots', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/sameday_sim', exist_ok=True)
os.makedirs('./results/decoding_and_cca', exist_ok=True)

### identify significantly modulated neurons ###
print('\n\ncomputing modulation')
sb.call(['python', 'compute_sameday_similarity.py'])

### run neural analyses for twotap (~20 mins) ###
print('\n\nrunning neural analyses for twotap task')
sb.call(['python', 'run_neural_analyses.py', '0'])

### run neural analyses for wds (~30 mins) ###
print('\n\nrunning neural analyses for WDS task')
sb.call(['python', 'run_neural_analyses.py', '1'])

### run latent space and decoding analyses ###
print('\n\nrunning latent stability')
sb.call(['python', 'latent_stability.py', '1'])

### run behav analyses for twotap ###
print('\n\nrunning behavioral analyses for twotap task')
sb.call(['python', 'run_behav_analyses.py', '0'])

### run behav analyses for WDS ###
print('\n\nrunning behavioral analyses for WDS task')
sb.call(['python', 'run_behav_analyses.py', '1'])

### run IPI analyses (~1 min)###
print('\n\nrunning IPI analyses')
sb.call(['python', 'analyze_ipis.py'])


