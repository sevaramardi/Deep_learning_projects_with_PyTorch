from main import test_loader
import matplotlib.pyplot as plt


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
for images, labels in test_loader:
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(classes[i])
    plt.show()
    break