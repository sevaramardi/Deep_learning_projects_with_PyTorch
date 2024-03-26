from main import test_loader
import matplotlib.pyplot as plt
images , labels = next(iter(test_loader))
names = ['cat', 'dog']
for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(images[i][0])
    plt.title(names[labels[i].item()])
plt.show()
