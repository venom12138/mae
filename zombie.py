import torch 
import time
import os
from pathlib import Path

_home = Path.home()
with open(f'{_home}/.core/flag', 'r+') as f:
    counter = eval(f.read())
    f.seek(0)
    # zombie 为0
    f.write(0)

a = torch.randn((34000,34000), device='cuda', dtype=torch.float64)
while True:
    a += torch.mul(a,a.T)
    time.sleep(0.3)

    with open(f'{_home}/.core/flag', 'r+') as f:
        identifier = f.read()[0]
    if identifier != 0:
        break