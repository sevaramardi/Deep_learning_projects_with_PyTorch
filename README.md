# Deep-learning-projects-with-PyTorch
#First repository Autoencoders
In this repository Autoencoders are used for image reconstruction task. There are two projects: CNN and ANN based Autoencoders.  

  -CNN based Autoencoders:
  
     -In this projects are bult classic CNN model with 2 layers convolution encoder which compress the data and 2 layers convolution transpose decoders which attempt to reconstruct the image. 
      
       - Used dataset CIFAR10.
       - Model is designed by using such as tools- Python, PyTorch, Jupyter Notebook, Numpy, Matplotlib.
       - Mean Square Error loss 
       - ReLU activation function in hidden layers, in output Tanh activation function.
       - Results are posted in the repository in Jupyter Nootebook. 
       - Code for plotting also available.
       
  -ANN based Autoenders:

    -In this projects are bult classic ANN model with 4 layers linear encoder which compress the data and 2 layers linear decoders which attempt to reconstruct the image. The encoder part           decreases number of neurons and decoders recunstruct image by increasing the 

       - Used dataset MNIST (Handwritten).
       - Model is designed by using such as tools- Python, PyTorch, Jupyter Notebook, Numpy, Matplotlib.
       - Mean Square Error loss 
       - ReLU activation function in hidden layers, in output Sigmoid activation function.
       - Results are posted in the repository in Jupyter Nootebook. 
       - Code for plotting also available.
      
#Second repository is built project for Correlation Coefficient as a condition Conditional GAN for Synthetic Crop Data: Solving Agriculture’s Data Availability and Quality Challenges
This is the group project of CTLab students. In this project we use CTGAN for creating crop data by setting new condition 'correlation coefficient' for increasing data quality.

    - Used dataset synthetic crop data
    - First model passes through data preprocessing:
        -label encoding
        -split data
        -converting to tensor
    - Set condition + noise, sends to generater.
    - Uses One generator and one discriminator.
    - Model is designed by using such as tools- Python, PyTorch, TensorFlow, Numpy, Matplotlib.
    - BSELoss Binary Square Loss
    - ReLU activation function and adam optimizer.

#Third repository is related to Image Classification task and there are 6 projects. First two classic image classification models for CIFAR10 and MNIST datasets, next comes the modified DenseNet, then Student-teacher model with modified AlexNet using depthwise-separaple convolution layers, Transfer learning using ResNet18, and last modified BAM. Used framework PyTorch.

    - CNN for CIFAR10:
      - 4 conv layers with BN for each layer, and pooling layer (used MaxPooling).
      - Dropout with probanilty of 0.25 for avoiding overfitting.
      - ReLU activation function and CrossEntropy loss.
      - Model is designed by using tools- Python, PyTorch, Numpy, Matplotlib.

    -CNN for MNIST(handwrite and fashion):
      - 2 conv layers, two poolings (avarage and max).
      - 3 linear layers.
      - ReLU activation function and CrossEntropy loss.
      - Model is designed by using tools- Python, PyTorch, Numpy, Matplotlib
      
    - Modified DenseNet:
      - Modified Dense block after input comes convolution layer for extracting more feautures, and there is added one extra convolution layer by making dense connecting more.
      - After each modofied block comes one more skip conection and Channel attention block from CBAM model, which lesrns 'what' in the image.
      - Model is designed by using tools- Python, PyTorch, Numpy, Matplotlib.
      - 7 conv layers in each block with k=4 growth rate, actv function relu, and cross entropy loss.
      - Channel attention block is light changed with Maxpoling as a downsampling method.
      - Architecture is provided.

    - Student-teacher model:
     - Modified AlexNet using depthwise-separable convolution layers instead of standard convolutions.
     - Teacher model is weights of original AlexNet.
     - Student model is modified ALexNet.
     - Model is designed by using tools- Python, PyTorch, Numpy, Matplotlib.
     - Architecture is provided.
     
    - Transfer Learning:
     - Pre-trained ResNet18 model is trained for classifying dogs.
     - used datased dogs and cats from Kaggle (https://www.kaggle.com/datasets/tongpython/cat-and-dog).


    - Modified BAM(BottleNeck attention models):
     - to BAM is added two extra blocks.
     - each block is devided into inner blocks wth same number conv, Bn layers and act functions.
     - Output of two inner blocks concantinates by elemnt-wise.
     - Model is designed by using tools- Python, PyTorch, Numpy, Matplotlib.
     - Architecture is provided.
     
#Fourth repository is Object detection. Now I am learning this field as it is one wildly used one. Here i am providing youtube channels from where I am learning it, for who curious and eager to learn Object detection.

     - DigitalScreene (333 lesson introduction to YOLOv8)
     - Aladdin Person (Object detection tutorial)
     - Nicolas Rennotte (Object detection with TensorFlow)
      
    
      
    






