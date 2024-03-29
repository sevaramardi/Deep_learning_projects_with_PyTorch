import os
import torch
import torchvision
import torch.nn as nn 
from torchvision import datasets, transforms

#data_dir = 'dog/data'
transform = {'train': transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.25,0.25, 0.25))])}

#train_data = ImageFolder(os.path.join(data_dir,'train'), transform['train'])
train_data = datasets.CIFAR10(root = './data', train=True, download = True, transform = transform['train'])
train_loader = torch.utils.data.DataLoader(dataset = train_data, shuffle=True, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# imges, labs = next(iter(train_loader))
# print(torch.min(imges), torch.max(imges))  #tensor(-2.) tensor(9)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #encoder
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)  #16
        self.relu =  nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32,3, stride=2, padding=1) #8
        self.bn2 = nn.BatchNorm2d(32)

        #decoder
        self.conv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1 ) #16
        self.conv4 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1 ) #32
        self.act = nn.Tanh()

    def forward(self, x):

        #encoder
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        #decoder
        out = self.relu(self.bn1(self.conv3(out)))
        out = self.relu((self.conv4(out)))
        out = self.act(out)
        return out * 2
    
model = CNN().to(device)
loss_ = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 25
outputs = []
for epoch in range(epochs):

    for images, labels in train_loader:

        images = images.to(device)
        #labels = labels.to(device)
        output = model(images)

        #forward
        loss_func = loss_(output, images)
        optimizer.zero_grad()
        loss_func.backward()
        optimizer.step()
     
    
    if epoch%2 == 0: 
        print(f" Epoch: {epoch+1}, loss: {loss_func.item():.4f}")
    outputs.append((images, output))
