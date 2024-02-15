import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
from main import train_loader, test_loader
from BAM import BAM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class BLOCK1(nn.Module):
    def __init__(self):
        super(BLOCK1, self).__init__()

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.relu = nn.ReLU()

        #Block1
        self.conv1 = nn.Conv2d(3,8,3, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3,stride=1, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)


        #Block2
        self.conv4 = nn.Conv2d(3,8,3, stride=1, padding = 1)
        self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(8,16,3,stride=1, padding = 1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16,32,3, stride=1, padding = 1)
        self.bn6 = nn.BatchNorm2d(32)

    def forward(self,x):

        #Block1
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.relu(self.bn3(self.conv3(out2)))
        #Block2
        out4 = self.relu(self.bn4(self.conv4(x)))
        out5 = self.relu(self.bn5(self.conv5(out4)))
        out6 = self.relu(self.bn6(self.conv6(out5)))

        pool1 = self.relu(self.pool1(out3))
        pool2 = self.relu(self.pool2(out6))
        out = torch.concat((pool1,pool2), dim=1)
        return out


class BLOCK2(nn.Module):
    def __init__(self):
        super(BLOCK2, self).__init__()
        

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.relu = nn.ReLU()

        #Block1
        self.conv1 = nn.Conv2d(3,64,3, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128 ,3,stride=1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128,3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)

        #Block2
        self.conv4 = nn.Conv2d(3,64,3, stride=1, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,128,3,stride=1, padding = 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,3, stride=1, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)

    def forward(self,x):

        #Block1
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.relu(self.bn3(self.conv3(out2)))
        #Block2
        out4 = self.relu(self.bn4(self.conv4(x)))
        out5 = self.relu(self.bn5(self.conv5(out4)))
        out6 = self.relu(self.bn6(self.conv6(out5)))

        pool1 = self.relu(self.pool1(out3))
        pool2 = self.relu(self.pool2(out6))
        out = torch.concat((pool1,pool2), dim=1)

        return out






class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.block1 = BLOCK1()
        self.block2 = BLOCK2()
   
        self.bam1 = BAM(gate_channel=64)
        self.bam2 = BAM(gate_channel=256)
        self.relu = nn.ReLU()
     
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4*4*256, 2000)
        self.bn = nn.BatchNorm1d(2000)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2000, 10)

    def forward(self,x):
        #first block
        out = self.block1(x) 
        out = self.bam1(out)
        
        
        #second block
        out = self.block2(x) 
        out = self.bam2(out)
        out = self.pool3(out)
        
       
        out = out.view(x.size(0),-1)
        out = self.fc2(self.drop(self.bn(self.fc1(out))))
       
        return out



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
num_epochs = 60

total_correct = 0
total_samples = 0
best_losses = {'train': [], 'val': []}
best_accs  = {'train': [], 'val': []}

for epoch in range(num_epochs):

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

model_path = 'path'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
  
