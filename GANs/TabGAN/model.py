import torch
import pandas as pd
import torch.nn as nn
from data_loader import dataloader

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class GENERATOR(nn.Module):
    def __init__(self):
        super(GENERATOR, self).__init__()
        self.model_gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,512),
            nn.ReLU(),
             nn.Dropout(0.2),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,29),
            nn.Softmax()
        )
        
    def forward(self,x):
     return self.model_gen(x)
    


class DISCRIMINATOR(nn.Module):
    def __init__(self, input_size):
        super(DISCRIMINATOR, self).__init__()
        self.model_dis = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.LeakyReLU(0.2),
            nn.Linear(150, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model_dis(x)

n_features = 29
generator = GENERATOR().to(device)
discrim = DISCRIMINATOR(n_features).to(device)

loss_ = nn.BCELoss()
g_optim = torch.optim.Adam(generator.parameters(), lr=0.002)
d_optim = torch.optim.Adam(discrim.parameters(), lr=0.002)
n_epochs = 1000 
#steps = len(data) #56717


for epoch in range(n_epochs):
    for batch in  dataloader:
        real_data = batch[0].to(device)
        #Train the Discriminator with real data
        d_optim.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1, device=device)
        real_output = discrim(real_data)
        d_loss_real = loss_(real_output, real_labels)

        # Train with fake data
        z = torch.randn(real_data.size(0), 100, device=device)  # Generate random noise
        fake_data = generator(z)
        fake_labels = torch.zeros(real_data.size(0), 1, device=device)
        fake_output = discrim(fake_data.detach())
        d_loss_fake = loss_(fake_output, fake_labels)
          
        # Compute total loss and update discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optim.step()

        # Training Generator
        g_optim.zero_grad()

        # Generate fake data
        fake_output = discrim(fake_data)
        g_loss = loss_(fake_output, real_labels)  # Trick the discriminator

        # Update generator
        g_loss.backward()
        g_optim.step()

    # Logging the 
    if d_loss.item() == 100.0:
        print(f'D became 100.0')
        exit()
    elif d_loss.item() == 0.0:
        print(f'D became 0.0')
        exit()
    elif g_loss.item() == 100.0:
        print(f'G became 100.0')
        exit()
    elif g_loss.item() == 0.0:
        print(f'G became 0.0')
        exit()
    else:
        print(f'Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

print('Training finished')
#print('Generating the synthetic data')


num_samples_to_generate = 1000
noise = torch.randn(num_samples_to_generate, 100, device=device) 
with torch.no_grad():
    generator.eval()
    synthetic_data = generator(noise).cpu().numpy()    
gen_data = pd.DataFrame(synthetic_data)
gen_data.to_csv('gen_data_crop.csv', index=False)
      


