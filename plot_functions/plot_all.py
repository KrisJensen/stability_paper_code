import subprocess as sb
import os

print('\nplotting all figures and printing results')

### make directories ###
print('\n\ncreating directories for results and figures')
os.makedirs('../paper_figs', exist_ok=True)
os.makedirs('../paper_figs/fig_intro', exist_ok=True)


##### main text figures #####

print('\n\nplotting schematics for introduction')
sb.call(['python', 'plot_intro_schematics.py'])

print('\n\nplotting RNN figure')
sb.call(['python', 'plot_rnn_fig.py'])

print('\n\nplotting experiment figure')
sb.call(['python', 'plot_exp_fig.py'])

print('\n\nplotting neural figure')
sb.call(['python', 'plot_neural_fig.py'])

print('\n\nplotting behavior figure')
sb.call(['python', 'plot_behav_fig.py'])

print('\n\nplotting WDS figure')
sb.call(['python', 'plot_wds_fig.py'])


##### supplementary figures #####

print('\n\nplotting supplementary RNN figure')
sb.call(['python', 'plot_supp_rnn.py'])

print('\n\nplotting supplementary behavior figure')
sb.call(['python', 'plot_supp_all_behav.py'])

print('\n\nplotting supplementary neural similarity figure')
sb.call(['python', 'plot_supp_all_neurons.py'])

print('\n\nplotting supplementary similarity by time figure')
sb.call(['python', 'plot_4C_supp.py'])

print('\n\nplotting supplementary decoding figure')
sb.call(['python', 'plot_supp_latent.py'])

print('\n\nplotting supplementary model fit and stability index figure')
sb.call(['python', 'plot_example_fits.py'])

print('\n\nplotting supplementary trimming figure')
sb.call(['python', 'plot_supp_trimmed.py'])

print('\n\nplotting supplementary fits by subsampled time figure')
sb.call(['python', 'plot_supp_fits_by_time.py'])

print('\n\nplotting supplementary behavioral similarity figure')
sb.call(['python', 'plot_supp_behav.py'])

print('\n\nplotting supplementary modulation figure')
sb.call(['python', 'plot_modulation_fig.py'])

