import os
import subprocess as sb

names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gorakh', 'Gandhar']

### warp all data ###
for wds in [False, True]:
    for name in names:
        if wds:
            name += '_wds'
        print('\nparsing', name)
        sb.call(['python', 'parse_rat.py', name]) #parse with warping
        sb.call(['python', 'parse_rat.py', name, 'trim']) #also parse with trimming!
