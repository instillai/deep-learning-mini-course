**************************************
Logistic Regression Code Documentation
**************************************

##################
##################
.. contents::
  :local:
  :depth: 5
  
----------------------------
Overview
----------------------------
Logistic regression is used to predict the categorical dependent variable with the help of one or more independent variables. The output of a logistic regression function can only be represented by a number between 0 and 1 and is represented by a sigmoid function.

.. figure:: ../_img/10.png

----------------------------
Step 0: Import libraries
----------------------------
Import the PyTorch libraries needed to train a logistic regression model. We also needed to import torchvision, 
which is a package that consists of popular datasets and image transformations for computer vision.

.. code:: python

  # Import all the packages
  import torch
  import torch.nn as nn
  import torchvision.datasets as dSets
  import torchvision.transforms as transforms
  
--------------------------------
Step 1: Create Dataset
--------------------------------
The first step of developing a logistic regression model is to find a dataset of interest. This can be created on your own
or can be loaded from any public data source. In this case, we chose to load the MNIST data set from the torchvision package. 
This dataset contains thousands of handwritten numbers from 1 to 9. Our goal is to train a logistic regression model that is 
able to decipher each handwritten digit.

.. code:: python

  train_dataset = dSets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
  test_dataset = dSets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

--------------------------------
Step 2: Make Dataset Iterable
--------------------------------
The next step is to make sure that we can iterate through the dataset that we just loaded. The PyTorch DataLoader function makes this step very simple.

.. code:: python

  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

---------------------------------
Step 3: Create Model Class
---------------------------------
Now, we need to create a class that defines the architecture of linear regression. This class is going to include the initialization of our model as well as a definition for our forward propagation. Forward propagation refers to the calculation and storage of intermediate variables (including outputs) for the neural network in order from the input layer to the output layer.

.. code:: python

  class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
      super(LogisticRegressionModel, self).__init__()
      self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
      y_predict = self.linear(x)
      return y_predict

-------------------------------------
Step 4: Instantiate Model Class
-------------------------------------
Next, we initialize the model class by importing our input and output dimensions into the logistic regression model. The input dimension is 784 because each image within the data set has 784 pixels (28*28). The output dimension is 10 because we are trying to determine what digit the handwritten picture depicts (0-9).

.. code:: python

  input_dim = 784
  output_dim = 10

  model = LogisticRegressionModel(input_dim,output_dim)

-------------------------------------
Step 5: Instantiate Loss Class
-------------------------------------
We then initialize the loss class by using CrossEntropyLoss to compute loss. CrossEntropyLoss measures the performance of a classification model whose output is a probability value between 0 and 1.

.. code:: python

  loss_fn = nn.CrossEntropyLoss()

-------------------------------------
Step 6: Instantiate Optimizer Class
-------------------------------------
The optimizer represents the learning algorithm that we have selected to use. In this case we have decided to use Stochastic Gradient Descent (SGD). 

.. code:: python

  learning_rate = 0.001
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

-------------------------------------
Step 6: Train Model
-------------------------------------
The final step in this linear regression is to train the regression model. During this step, we iterate through the images. As we move from image to image, we make the image a variable, clear unneeded parameters and then send the image into the logistic regression model as a parameter. From there, we are able to calculate loss using the loss function, call PyTorch’s back propagation function and update the parameters for the next image using the optimizer. Lastly, we provide a chunk of code that calculates an accuracy value and displays the loss and accuracy values for every hundredth iteration.

.. code:: python

  iter = 0
  for epoch in range(1):
      for i, (images, labels) in enumerate(train_loader):
          # Load images as Variable
          images = images.view(-1, 28*28).requires_grad_()
          labels = labels

          # Clear gradients w.r.t. parameters
          optimizer.zero_grad()

          # Forward pass to get output/logits
          outputs = model(images)

          # Calculate Loss: softmax --> cross entropy loss
          loss = loss_fn(outputs, labels)

          # Computes the sum of gradients of given tensors w.r.t. graph leaves
          loss.backward()

          # Updating parameters
          optimizer.step()

          iter += 1

          if iter % 100 == 0:
              # Calculate Accuracy         
              correct = 0
              total = 0
              # Iterate through test dataset
              for images, labels in test_loader:
                  # Load images to a Torch Variable
                  images = images.view(-1, 784).requires_grad_()

                  # Forward pass only to get logits/output
                  outputs = model(images)

                  # Get predictions from the maximum value
                  _, predicted = torch.max(outputs.data, 1)

                  # Total number of labels
                  total += labels.size(0)

                  # Total correct predictions
                  correct += (predicted == labels).sum()

              accuracy = 100 * correct / total

              # Print Loss
              print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

-------------------------------------
Example Output
-------------------------------------
This shows an output from one of our executions. The results display that after each set of 100 iterations, our logistic regression model becomes more accurate, which was the ultimate goal of the model.

Iteration: 100. Loss: 2.2325596809387207. Accuracy: 20

Iteration: 200. Loss: 2.1107139587402344. Accuracy: 33

Iteration: 300. Loss: 2.03490948677063. Accuracy: 47

Iteration: 400. Loss: 1.9995723962783813. Accuracy: 57

Iteration: 500. Loss: 1.884688138961792. Accuracy: 64

Iteration: 600. Loss: 1.8383146524429321. Accuracy: 68



-------------------------------------
References
-------------------------------------
- https://d2l.ai/chapter_multilayer-perceptrons/backprop.html
- https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
- https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/
- https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
- Image Source: logistic_regression.htm
- This tutorial is based off of the above links.


------
Code
------
.. _simpleLogCode: ../code/logisticregression.py
`Full Code <simpleLogCode_>`_ 
