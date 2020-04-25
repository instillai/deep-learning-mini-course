*********************
Batch Normalization
*********************

Many problems arise with deep neural networks. The more hidden layers a network has, the more complex training becomes. In a deep network, values are updated in each layer moving in reverse from output to input based on an estimated error value. The main problem here is that layers are updated before their preceding layers are updated. Batch Normalization solves this, providing a significant improvement to both speed and accuracy when training a deep network.

##################
##################
.. contents::
  :local:
  :depth: 3

---------------
Learning Rate
---------------
Batch normalization helps to create a process where multiple layers are updated in a fashion that is much more coordinated and synchronized. One way that batch normalization accomplishes this is through standardizing the activation values from preceding layers similar to how the input layer is standardized. This means taking a batch of data coming into a layer and shaping it to have a mean of 0 and deviation of 1 to help reduce the network’s inner covariate shift. The primary effect this activation data standardization is that it stabilizes the distribution of inputs during parameter updates while still allowing relationships between nodes to change. The figure below shows the functions used for normalization and is from https://arxiv.org/pdf/1502.03167v3.pdf

.. figure:: _img/normalization.JPG

-----------------------------
Effects
-----------------------------
The strategy of batch normalization in a deep neural network basically opens the door for layers to learn with less dependence on the workings of other layers. Batch normalization also allows for the use of relatively higher learning rates without the negative effects that it would previously cause. Using higher learning rates without a resulting accuracy and reliability decrease means faster training. It also means that some networks that wouldn’t train successfully due to learning rate could find success. Batch normalization shows regularization effects that allow for the reduction of overfitting which is discussed in the Regularization tutorial. These effects are most similar to those of Dropout, therefore much less Dropout is needed to produce similar results.

-----------
References
-----------

https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
