import gc
import torch

def empty_cache():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
