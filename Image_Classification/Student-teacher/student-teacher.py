import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from main import train_loader, test_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128,3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 128, 3)
        self.conv5 = nn.Conv2d(128,64, 3)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(7*7*64, 1200)
        self.bn4 = nn.BatchNorm1d(1200)
        self.fc2 = nn.Linear(1200,800)
        self.bn5 = nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800,10)
       

    def forward(self,x):
        x1 = self.relu(self.pool(self.conv1(x)))
        x1 = self.relu(self.bn2(self.conv2(x1)))
        x1 = self.relu(self.conv3(x1))
        x1 = self.relu(self.conv4(x1))
        x1 = self.relu(self.bn3(self.conv5(x1)))
   
        x2 = x1.view(-1, 7*7*64)
        x3 = self.relu((self.fc1(x2)))
        x4 = F.dropout(x3, p=0.25)
        x5 = self.relu(self.bn5(self.fc2(x4)))

        x7 = self.fc3(x5)
        return x7

teacher = Teacher().to(device)
teacher.load_state_dict(torch.load('./t1_.pt'))

class depthwise(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(depthwise,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.depthwise(x)
        x2 = self.pointwise(x1)
        return x2
    
class Student(nn.Module):
    def __init__(self):
        super(Student,self).__init__()
        self.conv1 = depthwise(3,64, kernel_size=5)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2,1)
        self.conv2 = depthwise(64,128, kernel_size=5)
        self.conv3 = depthwise(128, 256, kernel_size=5)
        self.conv4 = depthwise(256,128, kernel_size=5)
        self.conv5 = depthwise(128,32, kernel_size=5)
        self.bn0 = nn.BatchNorm2d(32)
        #self.conv4 = nn.Conv2d(128,32, 3)
        self.avg = nn.AvgPool2d(5,1)
        self.droup1  = nn.Dropout(0.5)
        self.layer1 = nn.Linear(6*6*32, 1000)
        self.layer2 = nn.Linear(1000, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.layer3 = nn.Linear(500, 10)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(self.pooling(x1)) #29x29
        x2 = self.conv2(x1)
        x2 = self.relu(self.pooling(x2)) #24x24
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = self.relu(self.bn0(self.conv5(x2)))
        x3 = self.avg(x2)  
        x3 = x3.view(-1, 6*6*32)
        x4 = self.relu(self.layer1(x3)) 
        x4 = self.relu(self.layer2(x4))   
        x4 = self.layer3(self.bn1(self.droup1(x4)))
       
        return x4
    
student = Student().to(device)

#weight initialization
def initialize_weights(model = student):
    #classname = model.__class__.__name__
    # fc layer
    if isinstance(model, nn.Linear):
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif isinstance(model, nn.BatchNorm2d) or isinstance(model, nn.BatchNorm1d):
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

student.apply(initialize_weights);


optimizer = torch.optim.Adam(student.parameters(), lr=0.01)


def distillation(y,labels,teacher_scores, T, alpha):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y/T, dim=-1), F.softmax(teacher_scores/T, dim=-1)) * (T*T * 2.0 + alpha) + F.cross_entropy(y,labels) * (1.-alpha)



loss_func = nn.CrossEntropyLoss()
num_epochs = 200
total_correct = 0
total_samples = 0
for epoch in range(num_epochs):

    n_correct = 0
    n_samples = 0
    student.train()
    step = len(train_loader)
    train_loss = 0
    epoch_loss = 0
    for i, (images,target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        
        #forwardpass
        output = student(images)
        teacher_output = teacher(images).detach()
        loss_train = distillation(output,target,teacher_output, T=15.0, alpha=0.7)
        
        
        #backwardpass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        #update the loss
       
        _, predicted = torch.max(output, 1)
        n_correct += (predicted == target).sum().item()
        n_samples += target.size(0)

      
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)

        train_acc = (100.0*n_correct)/n_samples
        if (i+1) % 700 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{step}], Loss [{loss_train.item():.4f}], Current_acc [{train_acc:.2f}%]')
            print('-'*10)

avarage_acc = (100*total_correct)/total_samples
print(f'Avarage accuracy: {avarage_acc}%')
print('Training finished')
model_path = 'path.pt'
torch.save(teacher.state_dict(), model_path)
print(f'Model saved to {model_path}')    






