*************************
Neural Network Classifier
*************************

##################
##################
.. contents::
  :local:
  :depth: 8

==========================
Overview
==========================
Neural networks are computing systems that are modeled after the brains of living animals. They consist of neurons that are all connected to create a network. These systems "learn" from data that is provided to them. By providing neural networks with enough data, they are capable of making accurate predictions by training and learning from the data.

==========================
The Basics
==========================
The function of neural networks is fairly straightforward and basic. An input is provided to the network, and after some calculations are made, an output is returned. For example, if we wanted to have a program that could determine if there is a dog in a picture then the input to the neural network would be a picture, and the output would be true or false based on whether the network thinks the picture contains a dog or not.

In order for neural networks to be able to make these predictions, they must be given data to train. Training a neural network involves providing a large amount of input data and it's corresponding output data. Neural networks are able to make calculations to infer relationships between the inputs and outputs, and create a system for accurately being able to predict an outcome when only given an input.

============================
Convolutional Neural Network
============================
The architecture that is behind neural networks is always fairly straightforward. While there are many different types of architectures 
that are used for getting more accurate predictions in specific scenarios, they all involve input nodes and an output node, or "neuron". The input neurons take data about a scenario, and multiple layers on the "inside" of the network calculate what the outcome will be. What makes a neural network "deep" is when there are more than a single layer of neurons between the input and output neurons, as can be seen below. 


===========================================
Code for a Simple Neural Network Classifier
===========================================
To begin writing code with the PyTorch library, it is important to ensure that you have imported torch at the beginning of your python program. In the following code snippet, we import torch neural network library as well as an optimizer for the neural network which will be explained in further detail in step 6. 

.. code:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

For this particular example, we will need to import the torchvision library for the data set as well.
The dataset consists of classes airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The images in CIFAR-10 are of size 3x32x32 . i.e. 3-channel color images of 32x32 pixels in size.

.. code:: python

    import torchvision
    import torchvision.transforms as transforms
    
In addition to the dataset we will need to import matplotlib and numpy to plot the images as well as use numpy for some computation.

.. code:: python
    
    import matplotlib.pyplot as plt
    import numpy as np

--------------------------------
Step 1: Data - CIFAR10
--------------------------------
Load and Nomralize CIFAR10 dataset. The following code block is a complete code block of this step. We will break it down further and explain each function and variable.

.. code:: python
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train = False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
The following line creates a transforms variable used to create training sets and data sets, it is a list of transforms which is created by transforms.Compose() function. The output of the torchivision dataset we are using in this example are PIL images of range [0, 1]. We use transforms.Compose() as a function to compose several image transformations together. Within the parameters of the Compose function, we state transforms.ToTensor() covert the PIL image to a tensor which is a generalization of vectors and matrices representing a multidimensional array which will be processed as the data input. transforms.Normalize() takes in Tensor image size (C, H, W) as a parameter and normalizes a tensor image with mean and standard deviation.

.. code:: python

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
Now we define training set and a test set with the CIFAR10 dataset. There are many data sets that could be used using the torchvision library such as MNIST, Flickr, USPS, KMNIST, and many more. For this example we use the CIFAR10 dataset explained earlier on this page. The parameters are root, train, transform, and download. \\

Root \\

train \\

transform \\
download \\

.. code:: python
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train = False, download=True, transform=transform)


    
The following code snippets will be functions that will get and plot some image files from the CIFAR10 dataset which we loaded in the code snippet shown above. In this snippet we will use numpy and matplotlib to show the images.

.. code:: python

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.show()
    
    # obtain some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    
The following images contain the console output if the code were to be run as of now.


---------------------------------------------
Step 2: Define a Convolutional Neural Network
---------------------------------------------
Our Convolutional Neural Network will take 3-channel images. This is where the torch.nn library will be used to define our neural network.

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np

.. code:: python

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
In this step, we will also define a forward propagation function within the neural network. 

.. code:: python

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
Finally, create an instance of your neural network.

.. code:: python
            
    net = Net()
    
    
    


--------------------------------------------
Step 3: Define a Loss Function and Optimizer
--------------------------------------------
In this step we define a loss function and an optimizer. A loss function as discussed in Logistic Regression, Backpropagation, and the Gradient Descent section will map values of one or more variables into a real number representing a cost to an event. In this code snippet we will use the CrossEntropyLoss.

And we define it like so..

.. code:: python

    criterion = nn.CrossEntropyLoss()
    
When defining our optimizer which will attempt to minimize loss, this is where the torch.optim libary comes into play. 

.. code:: python

    import torch.optim as optim

In this code snippet, we will use SGD which stands for Stochastic Gradient Descent.

And we define the optimizer like so..

.. code:: python

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
  

-------------------------------------
Step 4: Training the Network
-------------------------------------
At this point, we have defined our dataset, our Convolutional Neural Network, forward propagation, loss function, and optimizer. Therefore, we will train the neural network.

.. code:: python

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad() # Why?
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')

-------------------------------------
Step 5: Test the Network on Test Data
-------------------------------------


------------------------------------------
Step 6: Results
------------------------------------------



=============
References
=============
This tutorial was inspired by the tutorial provided at https://pytorch.org/docs/stable/torchvision/transforms.html created by 14 contributors, last contributed on October 13, 2019.  View contributors and contributions here: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

Additional Supplementary References: 

- https://pytorch.org/docs/stable/torchvision/transforms.html
- https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
- https://pytorch.org/docs/stable/torchvision/transforms.html
- 

=============
Code
=============
.. _nnClassCode: ../code/NNclassifier.py
`Full Code Steps 1 - 4 <nnClassCode_>`_

.. _nnClassTest: ../code/NNclassifier_test.py
`Full Code Step 5 <nnClassTest_>`_

=============
Next Section
=============
.. _reg: regularization.rst
`Next Section: More on Deep Neural Networks: Regularization <reg_>`_ 
