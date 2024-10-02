import gc
import torch

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
