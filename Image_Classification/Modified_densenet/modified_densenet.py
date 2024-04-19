import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
from main import train_loader, test_loader
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    

class BLOCK1(nn.Module):
    def __init__(self):
        super(BLOCK1, self).__init__()

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.relu = nn.ReLU()

        #Block1
        self.conv1 = nn.Conv2d(3,7,3, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(7)

        self.conv2 = nn.Conv2d(10,14,3,stride=1, padding = 1)
        self.bn2 = nn.BatchNorm2d(14)

        self.conv3 = nn.Conv2d(24,28,3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(28)

        #extra convs for usage
        self.conv4 = nn.Conv2d(52,56,3, stride=1, padding = 1)
        self.bn4 = nn.BatchNorm2d(56)

        self.conv5 = nn.Conv2d(108,112,3, stride=1, padding = 1)
        self.bn5 = nn.BatchNorm2d(112)



    def forward(self,x):

        #Block1
        out0 = self.relu(self.bn1(self.conv1(x))) # 1stconv
        out1 = torch.concat((out0,x),dim=1)  #10

        out2 = self.relu(self.bn2(self.conv2(out1))) #2conv
        out3 = torch.concat((out2,out0),dim=1)  
        out4 = torch.concat((out3,x),dim=1)  #24

        out5 = self.relu(self.bn3(self.conv3(out4))) #3conv #28
        out6 = torch.concat((out5,out0),dim=1) #
        out7 = torch.concat((out6,out2), dim=1) # 
        out8 = torch.concat((out7,x), dim=1) #52
        

        out9 = self.pool1(out8)  #[64,52,wxh]
        
       
        return out9


class BLOCK2(nn.Module):
    def __init__(self):
        super(BLOCK2, self).__init__()
        

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.relu = nn.ReLU()

        #Block1
        self.conv1 = nn.Conv2d(52,56,3, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(56)

        self.conv2 = nn.Conv2d(108,112 ,3,stride=1, padding = 1)
        self.bn2 = nn.BatchNorm2d(112)

        self.conv3 = nn.Conv2d(220,224,3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(224)

        #Block2
        self.conv4 = nn.Conv2d(3,64,3, stride=1, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,128,3,stride=1, padding = 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,3, stride=1, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)

    def forward(self,x):

        #Block1
        out0 = self.relu(self.bn1(self.conv1(x))) #7xwxh 1stconv
        out1 = torch.concat((out0,x),dim=1)  #108
        
        out2 = self.relu(self.bn2(self.conv2(out1))) #2conv
        out3 = torch.concat((out2,out0),dim=1)  
        out4 = torch.concat((out3,x),dim=1)  #220

        out5 = self.relu(self.bn3(self.conv3(out4))) #3conv #28
        out6 = torch.concat((out5,out0),dim=1) #
        out7 = torch.concat((out6,out2), dim=1) # 
        out8 = torch.concat((out7,x), dim=1) #444

        
        return out8


class BLOCK3(nn.Module):
    def __init__(self):
        super(BLOCK3, self).__init__()
        

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.relu = nn.ReLU()

        #Block1
        self.conv1 = nn.Conv2d(444,380,3, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(380)

        self.conv2 = nn.Conv2d(380,280 ,3,stride=1, padding = 1)
        self.bn2 = nn.BatchNorm2d(280)

        self.conv3 = nn.Conv2d(280,220,3, stride=1, padding = 1)
        self.bn3 = nn.BatchNorm2d(220)

        self.conv4 = nn.Conv2d(220,224,3, stride=1, padding = 1)
        self.bn4 = nn.BatchNorm2d(224)

        self.conv5 = nn.Conv2d(224,228,3,stride=1, padding = 1)
        self.bn5 = nn.BatchNorm2d(228)

        self.conv6 = nn.Conv2d(128,128,3, stride=1, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)

    def forward(self,x):

        #Block1
        out0 = self.relu(self.bn1(self.conv1(x))) #7xwxh 1stconv
        out1 = self.relu(self.bn2(self.conv2(out0))) #2conv
        out2 = self.relu(self.bn3(self.conv3(out1))) #3conv #220
        out3 = self.relu(self.bn4(self.conv4(out2))) #224
        out4 = self.relu(self.bn5(self.conv5(out3))) #228

        out5 = torch.concat((out4,out3),dim=1) #452
       
        
        return out5



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.block1 = BLOCK1()
        self.block2 = BLOCK2()
        self.block3 = BLOCK3()
        
        self.bam1 = CBAM(52)
        self.bam2 = CBAM(444)
        self.bam3 = CBAM(452)
        self.relu = nn.ReLU()
     
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4*4*452, 7300) # we can change the size of the image by using poolings several times
        self.bn = nn.BatchNorm1d(7300)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(7300, 10)

    def forward(self,x):
        #first pair block
        out = self.block1(x) # first modifiead denseBlock  
        out = self.bam1(out) # first channel block
        #second pair block
        out = self.block2(out) #second modifiead denseBlock  
        out = self.bam2(out) #second channel block
        #3 pair block
        out = self.block3(out) #third modifiead denseBlock  
        out = self.bam3(out)  #third channel block
        
        out = self.pool3(out) #
        out = self.pool3(out) #4x 
        
        out = out.view(x.size(0),-1)
        out = self.fc2(self.drop(self.bn(self.fc1(out))))
       
        return out



model = Network().to(device)


loss_ = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


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
num_epochs = 5

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

  
plt.title('Train-Val accuracy')
plt.plot(range(1,num_epochs+1), metric_history['train'], label='train')
plt.plot(range(1,num_epochs+1), metric_history['val'], label='valid')
plt.xlabel('Training Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.title('Train-Val  Loss')
plt.plot(range(1,num_epochs+1), loss_history['train'], label='train')
plt.plot(range(1,num_epochs+1), loss_history['val'], label='valid')
plt.xlabel('Training Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
