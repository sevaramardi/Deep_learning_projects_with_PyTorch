import torch
import torch.nn as nn
from main import train_loader,device , test_loader
import torch.nn.functional as F

class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()

        #1st layer
        self.conv1 = nn.Conv2d(3,32,3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()
        #2nd layer
        self.conv3 = nn.Conv2d(64,128,3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,3)
        self.bn4 = nn.BatchNorm2d(256)
        
       
        #3nd Flatten layer
        self.fc1 = nn.Linear(256*5*5,200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)


    def forward(self,x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.relu(self.drop(self.pool1(x2)))
        x4 = self.bn3(self.conv3(x3))
        x5 = self.bn4(self.conv4(x4))
        x6 = self.relu(self.drop(self.pool1(x5)))
        x7 = x6.view(-1,256*5*5)
        x8 = self.fc3(self.fc2(self.fc1(x7)))
        return x8
model = CIFAR10().to(device)
#training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
l = nn.CrossEntropyLoss()
steps = len(train_loader)
epochs = 200

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = l(outputs, labels)

        #backward passS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i+1}/{steps}], Loss [{loss.item():.4f}]')
print('Training finished')
 
#saving the model
model_path = 'C:/Users/CT/Desktop/projects/Cifar/cifar10_model_pynet_234.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

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

