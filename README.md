# Neural Network
For this project I created a neural network based on Chapter 11 of "Elements of Statistical Learning" (Hastie, Tibshirani, Friedman, 2008). The purposes of the project were to both gain a deeper understanding of neural nets and to demonstrate the ability to turn math on the page into functional code. I suggest viewing the regression notebook first, as it provides a more detailed description of the code and math.
# Regression
Although neural networks are more commonly associated with classification problems, they can also be used for regression. The output
is directly predicted by using the identity function rather than the softmax function, and there is only one "class," the predicted output.
# Classification
I used the classic Iris dataset (Fisher, 1936) to test the classification abilities of the neural network. The gradient equations are different than the regression case because of the use of the softmax function. The goal of the network is to correctly predict flower type based on petal length and width.
# ADAM
I implemented the ADAM stochastic optimization algorithm (Kingma and Ba, 2015) to practice translating math from an academic paper into working code. I tried out the solver on the same Iris dataset, and compare the results to the steepest descent solver in the ADAM notebook.
