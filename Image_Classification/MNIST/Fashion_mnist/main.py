import torch 
import torchvision
import torchvision.transforms as transforms
#print('Libreries imported')
#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5),(0.5))]
)
#downloading the data 
train_data = torchvision.datasets.FashionMNIST(root = './data',
                                              train = True,
                                              download = True,
                                              transform = transform
                                              )
test_data = torchvision.datasets.FashionMNIST(root = './data',
                                              train = False,
                                              transform = transform)
#loader
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           shuffle = True,
                                           batch_size = 6)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          shuffle = False,
                                          batch_size = 6)


