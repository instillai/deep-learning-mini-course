*************************
Neural Network Classifier
*************************

##################
##################
.. contents::
  :local:
  :depth: 7

==========================
Overview
==========================
As discussed in previous sections, neural networks are computing systems that are modeled after the brains of living animals. They consist of neurons that are all connected to create a network. These systems "learn" from data that is provided to them. By providing neural networks with enough data, they are capable of making accurate predictions by training and learning from the data.

In this section, we will discuss and implement a Neural Network Classifier. Neural Network Classifiers are neural networks that are trained to classify an input to an output defining the input as a class of an object. For our purposes, we will focus on the classification of images. 

============================
Convolutional Neural Network
============================
The type of Neural Network used in this section is a Convolutional Neural Network. Convolutional Networks have advantages over normal neural networks when it comes to image classifications. The problem with image classification is that images with many color channels and pixels becomes very hard to computationally define and train models. Convolutional Neural Networks transform images into forms that are easier to process by passing a filter over the original image, conducting matrix multiplication over a subsection of pixels in the original image. It repeats this process until all subsections of the original image has been processed. 

Once the original image has been processed, the Convolutional Neural Network will undergo the pooling process which will reduce the spatial size of the convoluted features. This reduction in complexity will drecrease the computational cost of analyzing the dataset. 

After the pooling process, the information from the original image will be compressed enough to be used in a neural network model. Then the Convolutional Neural Network acts like a regular neural network where the information is passed to a set of neurons which pass values to additional layers until it results in a final output.


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
Load and Nomralize CIFAR10 dataset

.. code:: python
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train = False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
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
- https://towardsdatascience.com/classification-using-neural-networks-b8e98f3a904f

=============
Code
=============
.. _nnClassCode: NNclassifier.py
`Full Code Steps 1 - 4 <nnClassCode_>`_

.. _nnClassTest: NNclassifier_test.py
`Full Code Step 5 <nnClassTest_>`_

=============
Next Section
=============
.. _reg: regularization.rst
`Next Section: More on Deep Neural Networks: Regularization <reg_>`_ 

