import torch
import torchvision
import torch.nn as nn
from main import train_loader, test_loader,device

#network
class CONVFASHIONMNIST(nn.Module):
    def __init__(self):
        super(CONVFASHIONMNIST,self).__init__()

        #first layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()

        #second layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(2,2)

        #fully connected network
        self.fc1 = nn.Linear(64*5*5, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 10)


    def forward(self,x):
        x1 = self.relu(self.pool1(self.bn1(self.conv1(x))))
        x2 = self.relu(self.pool2(self.bn2(self.conv2(x1))))
        x3 = x2.view(-1, 64*5*5)
        x4 = self.fc3(self.fc2(self.fc1(x3)))
        return x4

model = CONVFASHIONMNIST().to(device)

#training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

        #backward passS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i+1}/{steps}], Loss [{loss.item():.4f}]')
print('Training finished')
#PATH = './cnn.pth'
#torch.save(model.state_dict(), PATH)
