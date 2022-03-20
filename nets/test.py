from HUnet import HUnet
import torch
model = HUnet
x = torch.rand((512,512,3))
print(x.size())
output = model(x)
print(list(output.size))