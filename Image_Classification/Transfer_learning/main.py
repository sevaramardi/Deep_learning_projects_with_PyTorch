import os
import torch
import time
import torchvision
import torch.nn as nn
#from torch.optim import lr_schedular
from torchvision import datasets, models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ',device)

data_transformer1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.25, 0.25, 0.25))
])
data_transformer2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])
data_dir = 'data'

train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transformer1)
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size = 16)

test_data = datasets.ImageFolder(os.path.join(data_dir,'test'), data_transformer2)
test_loader = torch.utils.data.DataLoader(dataset=test_data, shuffle= False, batch_size=16)

#print(len(train_loader), len(train_data)) 501 8005
#print(len(test_loader), len(test_data)) 127 2023

