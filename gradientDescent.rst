-----------------
Gradient Descent
-----------------

.. figure:: _img/GradientDescentPic.png

The gradient descent algorithm is used to train input parameters on a set of data. The goal here is to find the optimal values of the parameters that result in a minimization of the cost function. Each iteration of gradient descent will bring the output of the cost function lower until reaching the minimal point.

**1 dimensional example**

| Consider parameter w and repeat the step:
| w := w - α dJ(w)/dw

The notation “:=” means update, α is the learning rate or how big of a step is taken on the iteration, dJ(w)/dw is the derivative or slope at the current point and represents the change to the parameter w. After the update, if the slope is positive then w will decrease, and if the slope is negative then w will increase. The graph of the function is convex therefore each update will bring you closer to the minimal point.

**2 dimensional example**

| Consider parameters w and b. You now need to update both parameters as such:
| w := w - α ∂J(w, b)/∂w
| b:= b - α ∂J(w, b)/∂b

What you are doing is updating w based on the partial derivative, or slope, in the w direction and updating b based on the partial derivative, or slope, in the b direction. When writing code, ∂J(w, b)/∂w is denoted as dw and ∂J(w, b)/∂b is denoted as db.
