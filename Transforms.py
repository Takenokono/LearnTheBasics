# This file is for learning `Learn The Basic` Pytorch tutorials
# This part learn about Transforms.


# TRANSFORMS

# all Torchvision datasets have tow parameters -transform to modify the fetures and target_transform to modify the labels.

# FashionMNIST features are in PIL images Format.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)