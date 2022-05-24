import torch 
import time
import os

a = torch.randn(1000,device='cuda')

while True:
    a += torch.dot(a,a.T)
    time.sleep(0.001)