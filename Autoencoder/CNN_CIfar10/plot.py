import matplotlib.pyplot as plt 
from cnn_autoencoder import outputs, epochs

for i in range(0, epochs):
    plt.figure(figsize=(9,2))
    plt.gray() 
    imgs = outputs[i][0].detach().cpu().numpy()
    out =  outputs[i][1].detach().cpu().numpy()

    for k, item in enumerate(imgs):
        if k >= 6: break
        plt.subplot(2,6, k+1)
        #item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    for j, item in enumerate(out):
        if j >= 6: break
        plt.subplot(2,6, 6+j+1)
        #item = item.reshape(-1,28,28)
        plt.imshow(item[0])
    plt.show()
