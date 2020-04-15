****************************
Neural Networks Architecture
****************************

##################
##################
.. contents::
  :local:
  :depth: 5

-------------------------
What is a neural network?
-------------------------
Neural networks are computing systems that are modeled after the brains of living animals. They consist of neurons that are all connected to create a network. These systems "learn" from data that is provided to them. By providing neural networks with enough data, they are capable of making accurate predictions by training and learning from the data.

-------------------------
The Basics
-------------------------
The function of neural networks is fairly straightforward and basic. An input is provided to the network, and after some calculations are made, an output is returned. For example, if we wanted to have a program that could determine if there is a dog in a picture then the input to the neural network would be a picture, and the output would be true or false based on whether the network thinks the picture contains a dog or not.

In order for neural networks to be able to make these predictions, they must be given data to train. Training a neural network involves providing a large amount of input data and it's corresponding output data. Neural networks are able to make calculations to infer relationships between the inputs and outputs, and create a system for accurately being able to predict an outcome when only given an input.

-------------------------
Architecture
-------------------------
The architecture that is behind neural networks is always fairly straightforward. While there are many different types of architectures 
that are used for getting more accurate predictions in specific scenarios, they all involve input nodes and an output node, or "neuron". The input neurons take data about a scenario, and multiple layers on the "inside" of the network calculate what the outcome will be. What makes a neural network "deep" is when there are more than a single layer of neurons between the input and output neurons, as can be seen below. 

.. figure:: _img/neuralnetwork.jpeg
