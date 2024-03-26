import torch 
import os
import time
import numpy as np
import torchvision
import torch.nn as nn 
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models,transforms
from main import test_loader, train_loader, train_data



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"working device is {device}")


def train_model(model, loss_, optimizer, scheduler, epochs=5):
    since= time.time()
    #for saving pt file
    #torch.save(model.state_dict(), 'best_model.pt')
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        runnig_loss = 0.0
        running_corss = 0
        n_samples = 0
        for images, labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            #forward pass
            phase = 'train'
            with torch.set_grad_enabled(phase=='train'):
                output = model(images)
                _,preds = torch.max(output,1)
                loss1 = loss_(output, labels)

                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

            running_corss += (preds==labels).sum().item()
            runnig_loss += loss1.item()*images.size(0)
            n_samples += labels.size(0)
        scheduler.step()

        epoch_loss = runnig_loss/len(train_data)
        epoch_acc = running_corss/ len(train_data)
        acc2 = running_corss*100/n_samples
        print(f"Len1: {len(train_data)} samples: {n_samples}")
        print(f" epoch Acc: {epoch_acc}, epoch Acc2: {acc2}")
        #print(f"Epoch: {epoch+1}/{epochs}, epoch_loss: {epoch_loss:.2f}, epoch_accuracy: {epoch_acc:.2f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
    
    time_el = time.time() - since
    print(f'Training completed in {time_el//60}minutes and {time_el%60}seconds and best_acc is {best_acc:.2f}')
    
    return model

model = models.resnet18(weights=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,2)

model = model.to(device)
loss_  = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, loss_, optimizer, step_lr_scheduler, epochs=5)

