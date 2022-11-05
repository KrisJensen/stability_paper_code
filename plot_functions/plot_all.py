import subprocess as sb
import os

print('\nplotting all figures and printing results')
pystr = 'python'

### make directories ###
print('\n\ncreating directories for results and figures')
os.makedirs('../paper_figs', exist_ok=True)
os.makedirs('../paper_figs/fig_intro', exist_ok=True)


##### main text figures #####

print('\n\nplotting Figure 1 schematics')
sb.call([pystr, 'plot_fig1_schematics.py'])

print('\n\nplotting Figure 2')
sb.call([pystr, 'plot_fig2.py'])

print('\n\nplotting Figure 3')
sb.call([pystr, 'plot_fig3.py'])

print('\n\nplotting Figure 4')
sb.call([pystr, 'plot_fig4.py'])

print('\n\nplotting Figure 5')
sb.call([pystr, 'plot_fig5.py'])

print('\n\nplotting Figure 6')
sb.call([pystr, 'plot_fig6.py'])


##### supplementary figures #####

print('\n\nplotting Extended Data Figure 1')
sb.call([pystr, 'plot_ext_fig1.py'])

print('\n\nplotting Extended Data Figure 2')
sb.call([pystr, 'plot_ext_fig2.py'])

print('\n\nplotting Extended Data Figure 3')
sb.call([pystr, 'plot_ext_fig3.py'])

print('\n\nplotting Extended Data Figure 4')
sb.call([pystr, 'plot_ext_fig4.py'])

print('\n\nplotting Extended Data Figure 5')
sb.call([pystr, 'plot_ext_fig5.py'])

print('\n\nplotting Extended Data Figure 6')
sb.call([pystr, 'plot_ext_fig6.py'])

print('\n\nplotting Extended Data Figure 7')
sb.call([pystr, 'plot_ext_fig7.py'])

print('\n\nplotting Extended Data Figure 8')
sb.call([pystr, 'plot_ext_fig8.py'])

print('\n\nplotting Extended Data Figure 9')
sb.call([pystr, 'plot_ext_fig9.py'])

print('\n\nplotting Extended Data Figure 10')
sb.call([pystr, 'plot_ext_fig10.py'])
