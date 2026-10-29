import torch

def check_cuda_state():
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available exiting !!!!")
