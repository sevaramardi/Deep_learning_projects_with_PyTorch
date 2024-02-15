import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from main import train_loader, test_loader
from spatial_attention import spatialAttention



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DenseLayer(nn.Module):
    def __init__(self,round, input_features, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        # Bottleneck layers
        self.bn1 = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        # Follow-up convolution1
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=1, stride=1, padding=1, bias=False)
        # Follow-up convolution2
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate*2)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, bn_size * growth_rate *2, kernel_size=1, stride=1, padding=1, bias=False)
        # Follow-up convolution3
        self.bn4 = nn.BatchNorm2d((bn_size * growth_rate)*2)
        self.conv4 = nn.Conv2d(bn_size * growth_rate*2, (bn_size * growth_rate)+ 142, kernel_size=3, stride=1, padding=1, bias=False)
        # Follow-up convolution4
        self.bn5 = nn.BatchNorm2d((bn_size * growth_rate)+ 142)
        self.conv5 = nn.Conv2d((bn_size * growth_rate)+ 142, bn_size * growth_rate*2, kernel_size=3, stride=1, padding=1, bias=False)
        # Follow-up convolution5
        self.bn6 = nn.BatchNorm2d((bn_size * growth_rate)*2)
        self.conv6 = nn.Conv2d( bn_size * growth_rate*2, input_features,  kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
     
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out = self.conv4(self.relu(self.bn4(out)))
        out = self.conv5(self.relu(self.bn5(out)))
        out = self.conv6(self.relu(self.bn6(out)))
        concat = torch.cat([x, out], 1)
        # Concatenate input feature map and output feature map along the channel dimension
        return concat


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            
            layers.append(DenseLayer(i, input_features + i * growth_rate, growth_rate, bn_size))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)
    


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dense_block = DenseBlock(num_layers=6, input_features=64, bn_size=4, growth_rate=32)
        self.attention = spatialAttention()
        self.fc1 = nn.Linear(4*4*256, 10)
        
 
    def forward(self,x):

        x1 = F.relu(self.conv1(x))
        #First_blocks dense and attention      
        x2 = self.dense_block(x1)
        x3 = self.attention(x2)
        x4 = self.pool1(x*x3)
        #Second_blocks dense and attention 
        x1 = F.relu(self.conv1(x4))
        x2 = self.dense_block(x1)
        x3 = self.attention(x2)
        x4 = self.pool1(x2*x3)
        #Third_blocks dense and attention 
        x1 = F.relu(self.conv2(x4)) 
        x2 = self.dense_block(x1)
        x3 = self.attention(x2)
        x4 = self.pool1(x2*x3)
        #Flatten_layer
        x5 = x4.view(x.size(0),-1)
        x6 = self.fc1(x5)
        return x6 
    

model = Network().to(device)


loss_ = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def val_accuracy(model):
    loss_func = nn.CrossEntropyLoss()
    running_loss = 0
    running_met = 0
    n_correct_val = 0
    n_samples_val = 0
    len_ = len(test_loader)
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        n_correct_val += (predicted == labels).sum().item()
        n_samples_val += labels.size(0)
        #running_met += (100.0*n_correct_val)/n_samples_val
        running_loss += loss_func(output,labels).item()

    loss_ = running_loss/len_
    metric = (100.0*n_correct_val)/n_samples_val
    return loss_, metric

loss_history = {'train': [], 'val': []}
metric_history = {'train': [], 'val': []}

best_loss_val = float('inf')
best_loss_train = float('inf')
best_acc_val = 0.0
best_acc_train = 0.0


step =  len(train_loader)
loss_func = nn.CrossEntropyLoss()
num_epochs = 20

total_correct = 0
total_samples = 0
best_losses = {'train': [], 'val': []}
best_accs  = {'train': [], 'val': []}

for epoch in range(num_epochs):
    #current_lr = get_lr(optimizer)
    n_correct = 0
    n_samples = 0
    epoch_loss = 0
    model.train()

    
    for i, (images,target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        
        #forwardpass
        output = model(images)
        loss_train = loss_func(output, target)
        
        #backwardpass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        #update the loss
        epoch_loss += loss_train.item()
        _, predicted = torch.max(output, 1)
        n_correct += (predicted == target).sum().item()
        n_samples += target.size(0)

      

    train_loss = epoch_loss/len(train_loader)
    train_acc = (100*n_correct)/n_samples
    loss_history['train'].append(train_loss)
    metric_history['train'].append(train_acc)
    model.eval()

    with torch.no_grad():
        val_loss, val_metric = val_accuracy(model)
    loss_history['val'].append(val_loss)
    metric_history['val'].append(val_metric)

    if train_loss < best_loss_train:
        best_loss_train = train_loss

        best_losses['train'].append(best_loss_train) 
    if val_loss < best_loss_val:
        best_loss_val = val_loss
        
        best_losses['val'].append(best_loss_val) 
    if train_acc > best_acc_train:
        best_acc_train = train_acc
        best_accs['train'].append(best_acc_train)

    if val_metric > best_acc_val:
        best_acc_val = val_metric
        best_accs['val'].append(best_acc_val)

    if epoch%1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train_acc [{train_acc:.2f}%], Val_acc [{val_metric:.2f}%]: Train_loss [{train_loss:.4f}], Val_loss [{val_loss:.4f}]' )
        print('-'*10)



b_train_acc = max(best_accs['train'])
val_b_acc = max(best_accs['val'])
print(f'train: {b_train_acc}, Val: {val_b_acc}')
b_train_l = min(best_losses['train'])
b_val_l = min(best_losses['val'])
print(f'Train loss: {b_train_l}, val loss: {b_val_l}')

print('Training finished')

# model_path = 'C:/Users/CT/Desktop/projects/cifa10/attention.pt'
# torch.save(model.state_dict(), model_path)
# print(f'Model saved to {model_path}')
  
