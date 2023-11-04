import torch 
import torchvision
import torchvision.transforms as transforms
#print('Librerias were imported succesfully')

#device will use GPU if it is available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters for neural network
input_size = 784
hidden_size = 200
classes = 10
epochs = 2
batch_size = 100
lr = 0.001


#Downloading the MNIST data from torchvision base
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
