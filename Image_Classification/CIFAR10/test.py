from main import test_loader, device
from model import model
from main import device, test_loader
import torch 

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _,predicted = torch.max(outputs, 1)
      
        n_samples +=labels.size(0)
        n_correct += (predicted == labels).sum().item()
#
    acc = 100.0*n_correct/n_samples
    print(f'Accuracy: {acc}%')
