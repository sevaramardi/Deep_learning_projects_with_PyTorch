import torch 
import torch.nn as nn
import torchvision
import numpy 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_data = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_data, shuffle=True, batch_size = 64)

test_data = datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())
test_loader = DataLoader(dataset = test_data, shuffle = False, batch_size = 64)

images, labels = next(iter(test_loader))
#print(torch.min(images), torch.max(images))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class linear_autoencoder(nn.Module):
    def __init__(self):
        super(linear_autoencoder,self).__init__()
        #encoder
        self.fc1 = nn.Linear(28*28, 600)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200,100)
        

        #decoder
        self.fc5 = nn.Linear(100, 200)
        self.fc6 = nn.Linear(200, 400)
        self.fc7 = nn.Linear(400, 600)
        self.fc8 = nn.Linear(600, 28*28)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        #encoder
        x = x.view(-1,28*28)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))

        #decoder
        out = self.relu(self.fc5(out))
        out = self.relu(self.fc6(out))
        out = self.relu(self.fc7(out))
        out = self.sigmoid(self.fc8(out))
        return out
    
model = linear_autoencoder().to(device)

loss_ = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
steps = len(train_loader)
epochs = 4
samples = len(train_data)
outs = []

for epoch in range(epochs):
    n_correct= 0 
    n_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        #labels = labels.to(device)
        output = model(images)
        images = images.reshape(-1,28*28)
        
        #forward 
        loss = loss_(output,images)
        #backward
        optim.zero_grad()
        loss.backward() 
        optim.step()

    outs.append((images,output))
    print(f"Epoch: {epoch+1}, epoch_loss: {loss.item():.4f}")   
 
   
    
    

        
    


