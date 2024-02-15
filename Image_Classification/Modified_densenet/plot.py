import matplotlib.pyplot as plt
from modified_densenet import num_epochs, metric_history,loss_history


plt.title('Train-Val accuracy')
plt.plot(range(1,num_epochs+1), metric_history['train'], label='train')
plt.plot(range(1,num_epochs+1), metric_history['val'], label='valid')
plt.xlabel('Training Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.title('Train-Val  Loss')
plt.plot(range(1,num_epochs+1), loss_history['train'], label='train')
plt.plot(range(1,num_epochs+1), loss_history['val'], label='valid')
plt.xlabel('Training Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()