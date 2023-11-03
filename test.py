from model import model, device
from main import train_loader, test_loader
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        
    acc = 100.0*n_correct / n_samples
    print(f'Accuracy: {acc}%')
    
