from main import train_loader, test_loader, device
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

#hyper parameters for neural network
input_size = 784
hidden_size = 200
classes = 10
epochs = 2
batch_size = 100
lr = 0.001


# =============================================================================
#I uncommented this line , but it can be used for vizualization of data
# examples = iter(train_loader)
# example_data, example_feature = next(examples)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()
# =============================================================================

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, classes):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, classes)
        
    def forward(self, x):
        l1 = self.relu(self.l1(x))
        l2 = self.l2(l1)
        return l2
    
    
model = NeuralNetwork(input_size, hidden_size, classes).to(device)


#Loss and optimizer 
l = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#train the model
steps = len(train_loader)
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        #forward pass
        outputs = model(images)
        loss = l(outputs,labels)
    
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# =============================================================================
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{steps}], Loss: {loss.item():.4f}')
# =============================================================================













