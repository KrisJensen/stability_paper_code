import subprocess as sb
import os

pystr = 'python'

### create directories ###
print('\n\ncreating directories for results')
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/sameday_sim', exist_ok=True)
os.makedirs('./results/decoding_and_cca', exist_ok=True)

### identify significantly modulated neurons ###
print('\n\ncomputing modulation')
sb.call([pystr, 'compute_sameday_similarity.py'])

### run neural analyses for twotap (~20 mins) ###
print('\n\nrunning neural analyses for twotap task')
sb.call([pystr, 'run_neural_analyses.py', '0'])

### run neural analyses for wds (~30 mins) ###
print('\n\nrunning neural analyses for WDS task')
sb.call([pystr, 'run_neural_analyses.py', '1'])

# ### run latent space and decoding analyses ###
print('\n\nrunning latent stability and decoding models')
sb.call([pystr, 'latent_stability.py'])
sb.call([pystr, 'fit_encoding_model.py'])
sb.call([pystr, 'crossval_encoding.py'])

# ### run behav analyses for twotap ###
print('\n\nrunning behavioral analyses for twotap task')
sb.call([pystr, 'run_behav_analyses.py', '0'])

### run behav analyses for WDS ###
print('\n\nrunning behavioral analyses for WDS task')
sb.call([pystr, 'run_behav_analyses.py', '1'])

# ### run trimming analyses ###
print('\n\nrunning analyses with trimming')
sb.call([pystr, 'run_trimmed_analyses.py', '0'])
sb.call([pystr, 'run_trimmed_analyses.py', '1'])

# ### run IPI analyses (~1 min)###
print('\n\nrunning IPI analyses')
sb.call([pystr, 'analyze_ipis.py'])


