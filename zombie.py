import torch 
import time
import os

a = torch.randn((34000,34000), device='cuda', dtype=torch.float64)

while True:
    a += torch.mul(a,a.T)
    # time.sleep(0.001)