import torch
import torchvision
import torchvision.transforms as transforms

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]


transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),  # Padding and then cropping to the original size
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
testset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform = transforms_train)

train_loader = torch.utils.data.DataLoader(dataset = trainset, shuffle= True, batch_size = 64)
test_loader = torch.utils.data.DataLoader(dataset = testset, shuffle=False, batch_size = 64) 