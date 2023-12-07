import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

#transform of data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#data
train_data = torchvision.datasets.CIFAR10(root = './data',
                                          train = True,
                                          download = True,
                                          transform = transform)
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           shuffle = True,
                                           batch_size = 4)
test_data = torchvision.datasets.CIFAR10(root = './data',
                                         train = False,
                                         transform = transform)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          shuffle = False,
                                          batch_size = 4)
