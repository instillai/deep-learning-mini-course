---------------------
Activation Functions
---------------------


Activation functions are used in deep learning between the layers of a neural network to build correlations between inputs and outputs. The higher the output of the activation functions for a neuron the more weight is added to the neuron and is passed to the next layer. Activation functions are used to train the layers in neural networks. There are many different activation functions that can be used which are shown in the figure below.


.. figure:: _img/activationFunctions.png



Different activation functions could also be used at the same time, for example tan h for activation in hidden layers and sigmoid for deactivation. The most common activation function (especially in the hidden layer) is ReLU (Rectified linear Unit) denoted by a = max(0, z). The ReLU function is most common because it is the most efficient for computing because it reduces the problem of a vanishing gradient. Leaky ReLU fixes the problem with the normal ReLU function that the slope if 0 when Z is negative by having a slightly negative slope when z is negative. Leaky ReLU is often slightly better than the normal ReLU function however the regular ReLU function is used more in practice.

-------
Sources
-------
| https://www.coursera.org/
| https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044 (image)
