# Deep-learning-projects-with-PyTorch
#First repository Autoencoders
In this repository Autoencoders are used for image reconstruction task. There are two projects: CNN and ANN based Autoencoders.  

  -CNN based Autoencoders:
  
     -In this projects are bult classic CNN model with 2 layers convolution encoder which compress the data and 2 layers convolution transpose decoders which attempt to reconstruct the image. 
      
       *Used dataset CIFAR10.
       *Model is designed by using such as tools- Python, PyTorch, Jupyter Notebook, Numpy, Matplotlib.
       *Mean Square Error loss 
       *ReLU activation function in hidden layers, in output Tanh activation function.
       *Results are posted in the repository in Jupyter Nootebook. 
       *Code for plotting also available.
       
  -ANN based Autoenders:

    -In this projects are bult classic ANN model with 4 layers linear encoder which compress the data and 2 layers linear decoders which attempt to reconstruct the image. The encoder part           decreases number of neurons and decoders recunstruct image by increasing the 

       *Used dataset MNIST (Handwritten).
       *Model is designed by using such as tools- Python, PyTorch, Jupyter Notebook, Numpy, Matplotlib.
       *Mean Square Error loss 
       *ReLU activation function in hidden layers, in output Sigmoid activation function.
       *Results are posted in the repository in Jupyter Nootebook. 
       *Code for plotting also available.
      
#Second repository is built project for Correlation Coefficient as a condition Conditional GAN for Synthetic Crop Data: Solving Agriculture’s Data Availability and Quality Challenges
This is the group project of CTLab students. In this project we use CTGAN for creating crop data by setting new condition 'correlation coefficient' for increasing data quality.

    *Used dataset synthetic crop data
    *First model passes through data preprocessing:
      -label encoding
      -split data
      -converting to tensor
    *Set condition + noise, sends to generater.
    *Uses One generator and one discriminator.
    *Model is designed by using such as tools- Python, PyTorch, TensorFlow, Numpy, Matplotlib.
    *BSELoss Binary Square Loss
    *ReLU activation function and adam optimizer.
    
    
    






