# Ttests if there is Cuda support and if pyTorch is correctly installed

import torch

print("Cuda available: ", torch.cuda.is_available())
x = torch.rand(5, 3)
print(x)
