********
Dropout
********

**Dropout** is another form of regularization that is often used to solve the problem of overfitting in deep learning. In dropout the model gets set with some probability value that will be applied to neurons in the neural network to determine if they will “fire” or not. For example a model trained with a probability of .5 will only train with half of the neurons will  fire. A visualization of this is shown below. 

.. figure:: _img/overfitting.PNG

Dropout can be applied to just one or all hidden layers of the network as well as visible and input layers however is never done on the output layer. Dropout works because with less neurons being used to apply weight, the overall complexity of the network is reduced which helps to solve overfitting. Dropout is only used when training the model and the neurons which get turned off change with every iteration, dropout is not used in the model after training for predictions. 
