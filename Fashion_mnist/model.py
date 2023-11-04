import torch
import torchvision 
import torch.nn as nn
from main import device, train_loader

class CONVMNIST(nn.Module):
    def __init__(self):
        super(CONVMNIST, self).__init__()
        # first layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
    
        #second layer
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
        # fully connected layer
        self.fc1 = nn.Linear(64*6*6, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
      layer1 = self.relu(self.pool1(self.conv1(x)))
      layer2 = self.relu(self.pool2(self.conv2(layer1)))
      layer3 = layer2.view(-1, 64*6*6)
      layer4 = self.relu(self.fc1(layer3))
      layer5 = self.relu(self.fc2(layer4))
      layer6 = self.fc3(layer5)
      return layer6

model = CONVMNIST().to(device)

# Trainig

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
l = nn.CrossEntropyLoss()
steps = len(train_loader)
epochs = 50
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = l(outputs, labels)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Step [{i+1}/{steps}] Loss: [{loss.item():.4f}]')
        
        
    