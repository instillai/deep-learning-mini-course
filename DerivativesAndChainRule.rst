-------------------------------
Derivatives and the Chain Rule
-------------------------------

##################
##################
.. contents::
  :local:
  :depth: 2


The **derivative** of a function is an equation to find the slope of a function at a given point. With a linear function, the slope is the same at every point and can be easily calculated. For example, consider the function y = 5x + 4. The derivative is the change in y divided by the change in x. Looking at x = 1 and x = 2, you can see that y = 9 and 14 respectively. In this case, when x increases by 1, y increases by 5. So dy/dx is 5. There are many derivative rules out there for all types of functions but the most important rule for deep learning is the chain rule.

The **chain rule** can be used to calculate derivatives of the cost function through backpropagation which means calculating the derivatives in chunks from right to left. When deriving an equation with multiple variables in deep learning, you will want to keep grouping parts of the equation together until arriving at a single group. The next step is to calculate derivatives starting at the single group working backwards down the group expansions, multiplying each groups derivative into the total as you go. For example, to find dJ/da you may have to have calculated dJ/dv * dv/du * du/da which equals dJ/da algebraically.


-----------
References
-----------
https://www.coursera.org/


-------------
Next Section
-------------
.. _Grad: gradientDescent.rst
`Next Section: Gradient Descent <Grad_>`_ 
