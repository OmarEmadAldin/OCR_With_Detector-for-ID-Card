import torch
def check_device():
    # Check if CUDA is available and return the appropriate device
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")