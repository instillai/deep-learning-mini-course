*********************
Gradient Setbacks
*********************

The basis behind most deep learning algorithms consists of using a gradient to minimize loss through optimizing parameters. While gradients can have groundbreaking success, they also have some limitations and setbacks.

##################
##################
.. contents::
  :local:
  :depth: 2

---------------
Learning Rate
---------------
.. figure:: _img/setbacks.png

When optimizing functions using gradients, the goal is to find parameter values that converge at a point where the function’s loss is equal to or extremely close to the minimum. The learning rate parameter is what decides how much the initial parameters are altered when performing an iteration of gradient descent. Choosing this learning rate creates a prominent setback relating to convergence. If the chosen learning rate is fast, the iterations could end up skipping over the global minimum thus arriving at a local minimum. A fast learning rate basically trades off time for accuracy. On the contrary, choosing a slower learning rate doesn’t guarantee success. With a slow learning rate, the values may never converge because it would take more iterations than possible.


-----------------------------
Possible Solutions
-----------------------------
Strategies pertaining to adaptive learning rates can be applied to gradient descent to improve time and accuracy. There are two solid ways to adjust the iteration process to decrease the likelihood of problems caused by the learning rate. One is to adjust the learning rate proportionally to the size of the dataset. Depending on the chosen cost function, larger training sets can lead to larger steps per iteration. Having a learning rate that is adjusted to be smaller when training larger datasets can be a fix here. Another way to improve results by adapting the learning rate is to implement an algorithm that adjusts the learning rate after every iteration. This strategy accomplishes both time and accuracy improvements by moving faster in the beginning and gradually slower while approaching optimal parameter values. Although these strategies can produce significant improvements in gradient descent algorithms, they don’t eliminate all setbacks.


**Sources**

https://towardsdatascience.com/why-gradient-descent-isnt-enough-a-comprehensive-introduction-to-optimization-algorithms-in-59670fd5c096

http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/

https://hackernoon.com/gradient-descent-aynk-7cbe95a778da
