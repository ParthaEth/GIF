import sys
sys.path.append('../')
from model import ImgEmbedding
import torch

embd_frozen = ImgEmbedding(5, 3)
for params in embd_frozen.parameters():
    print(params)

print(embd_frozen(torch.tensor([1, 2], dtype=torch.long)))