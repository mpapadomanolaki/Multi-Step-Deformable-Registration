import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = 0
def to_cuda(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v
