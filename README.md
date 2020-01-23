# Neural Network
For this project I created a neural network based on Chapter 11 of "Elements of Statistical Learning" (Hastie, Tibshirani, Friedman, 2008). The purposes of the project were to both gain a deeper understanding of neural nets and to demonstrate the ability to turn math on the page into functional code. I suggest viewing the regression notebook first, as it provides a more detailed description of the code and math.
# Description
This network uses one hidden layer with an adjustable number of hidden units as well as adjustable learning rate and weight decay. The network was used for regression, so there is only one class and the network outputs values, rather than probabilities as would be the case for a classification problem. A penalty is added to the error function to limit the size of weights. Back-propagation is used to find optimal weights.
# Regression
Although neural networks are more commonly associated with classification problems, they can also be used for regression. The output
is directly predicted by using the identity function rather than the softmax function, and there is only one "class," the predicted output.
# Classification
I used the classic Iris dataset (Fisher, 1936) to test the classification abilities of the neural network. The gradient equations are different than the regression case because of the use of the softmax function. The goal of the network is to correctly predict flower type based on petal length and width.

