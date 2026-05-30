import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Available:", torch.cuda.is_available())