#!/usr/bin/python3

import os

process = [i.strip().split(' ') for i in os.popen('ps -A').read().split('\n')]
process = [i for i in process if ((len(i) > 3) and (i[-1] != '<defunct>'))]
process = [i for i in process if (i[-1] in ['interpreter', 'plasma-store-se'])]
process = [i[0] for i in process]
cmd = 'kill -9 ' + ' '.join(process)
cmd += ' >/dev/null'
print(cmd)
os.system(cmd)