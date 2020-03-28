# MNIST Digit CLassifier #
This file contains the following code:
1. MNIST-ANN.py: This script has the code to run an Artificial Neural Network for digit classification.
2. cnn_funcs.py: This script has the functions that I used to create the Convolution Neural Network. The functions are given below:
	- load_data(): It loads the mnist dataset from the keras library and preprocesses them to the right format.
	- model_gen(): This function creates the CNN architecture using the keras library.
	- evaluate_model(): This function fits the neural network using the training data. It also gives the confusion matrix and classification report.
	- test_accuracy(): This function predicts the output for the test set and gives the test accuracy.
3. MNIST-CNN.py: This script runs the neural network using the functions defined in cnn_funcs.py. The final weights of the neural network is saved as "model.h5".
4. model.h5: The saved neural network architecture with the associated weights.
5. GUI.py: This script defines some classes to create a Graphical User Interface which provides a window to draw a number. Then, the saved neural network model is used to classify the drawn digit.

## GUI for Classifying digits ##
The GUI is made using tkinter, opencv and PIL. So, to run the script these must be installed in your system. Because of problems between opencv and mac os, this GUI only works in Windows OS.

The GUI looks like this:

![](https://github.com/HRishabh11/MNIST/blob/master/img1.PNG)
![](https://github.com/HRishabh11/MNIST/blob/master/img2.PNG)
