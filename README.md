#Linear Regression with Multiple Variables in TensorFlow
In Andrew Ng shows how to generalize linear regression with a single variable to the case of multiple variables. Andrew Ng introduces a bit of notation to derive a more succinct formulation of the problem. Namely, n features x_1….x_n are extended by adding feature x_0which is always set to 1. This way the hypothesis can be expressed as:

 

For m examples, the task of linear regression can be expressed as a task of finding vector θ such that


 

is as close as possible to some observed values y_1,y_1,…..,y_m. The “as close as possible” typically means that the mean sum of square errors between  h_θ (x^((i)))  and y_i for i∈[1,m] is minimized. This quantity is often referred to as cost or loss function:

 


To express the above concepts in TensorFlow, and more importantly, have TensorFlow find θ that minimizes the cost function, we need to make a few adjustments. We rename vector θ, as w. We are not using x_0=1. Instead, we use a tensor of size 0 (also known as scalar), called b to represent x_0 . As it is easier to stack rows than columns, we form matrix  X, in such a way that the i-th row is the i-th sample. Our formulation thus has the form

 

This leads to the following Python code:

 

We first introduce a tf.placeholder named X_in. This is how we supply data into our model. Line 2 creates a vector w corresponding to θ.Line 3 creates a variable b corresponding to θ . Finally, line 4 expresses function h as a matrix multiplication of X_in and w plus scalar b

 

To define the loss function, we introduce another placeholder y_in. It holds the ideal (or target) values for the function h. Next, we create a loss_op. This corresponds to the loss function. The difference is that, rather than being a function directly, it defines for TensorFlow operations that need to be run to compute a loss function. Finally, the training operation uses a gradient descent optimizer, that uses learning rate of 0.3, and tries to minimize the loss.
Now we have all pieces in place to create a loop that finds w and b that minimize the loss function.

 

