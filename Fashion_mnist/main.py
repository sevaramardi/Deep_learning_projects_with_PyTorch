import torch 
import torchvision
import torchvision.transforms as transforms
print('Librerias imported succesfully')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#hyper parameters
input_size = 784
hidden_size = 200
classes = 10
epochs = 2
batch_size = 100
lr = 0.001

#MNIST data
train_data = torchvision.datasets.MNIST(root = './data',
                                        train = True,
                                        download= True,
                                        transform = transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root = './data',
                                       train = False,
                                       transform = transforms.ToTensor())

#loaders 
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           batch_size= batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size= batch_size,
                                          shuffle = False)

