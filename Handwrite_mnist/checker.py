#import matplotlib.pyplot as plt
import torch
from Handwrite_mnist.model import model
from Handwrite_mnist.main import device
#from test import predicted

from Handwrite_mnist.main import test_loader


exam = iter(test_loader)
images,labels = next(exam)
images = images.reshape(-1, 28*28).to(device)
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
        
print('Original:')
with torch.no_grad():

    labs = iter(test_loader)
    images, labels = next(labs)

    for label in labels[:10]:
        label = label.to(device)
        print(label.item(), end=" ")
    print()
    print('Predicted:')
    for pred in predicted[:10]:
        print(pred.item(), end=" ")
        
