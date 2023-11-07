import matplotlib.pyplot as plt
from main import test_loader

for images, labels in test_loader:
    for i in range(6):
        plt.subplot(2,3, i+1)
        plt.imshow(images[i][0], cmap='viridis')
    plt.show()
    break