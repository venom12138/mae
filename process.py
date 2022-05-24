import time
import os
from pathlib import Path

_home = Path.home()
with open(f'{_home}/.core/flag', 'r+') as f:
    counter = eval(f.read())
    f.seek(0)
    # 我的为1
    f.write('1')
    
time.sleep(1)
