

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as tfs
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from torch.utils.tensorboard import SummaryWriter

import os
import json
from PIL import Image

class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128, bias=False)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.bn1 = nn.BatchNorm1d(128)
        #self.bn2 = nn.BatchNorm1d(64)
        
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x)) 
        #x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x        

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(28*28, 128, bias=False),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 10)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

model = TestModel()
print(model)
