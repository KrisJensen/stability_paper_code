import os
import subprocess as sb
import sys

names = ['Hindol', 'Dhanashri', 'Jaunpuri', 'Hamir', 'Gorakh', 'Gandhar']        

pystr = 'python'

trim = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'trim':
        trim = True

### warp all data ###
for wds in [False, True]:
    for name in names:
        if wds:
            name += '_wds'
        print('\nparsing', name)

        if trim:
            sb.call([pystr, 'parse_rat.py', name, 'trim'])
        else:
            sb.call([pystr, 'parse_rat.py', name])


