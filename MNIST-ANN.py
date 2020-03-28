#Hand written digit classification using ANN

#importing packages
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

#importing the data
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")
X_train = train_data.iloc[:,1:785].to_numpy()
y_train = train_data.iloc[:,0].to_numpy()
X_test = test_data.iloc[:,1:785].to_numpy()
y_test = test_data.iloc[:,0].to_numpy()

#Encoding the labels
y_train = keras.utils.to_categorical(y_train,num_classes= 10)
#y_test = keras.utils.to_categorical(y_test,num_classes = 10)

#visualizing one of the digits
print(y_train[7])
plt.imshow(X_train[7,:].reshape(28,28)) #this image is of the digit 4

#standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creating the ANN
classifier = Sequential()
classifier.add(Dense(output_dim = 100, init = "uniform", activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = "uniform", activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = "uniform", activation = 'softmax'))
classifier.compile(optimizer = 'adam',loss = "categorical_crossentropy", metrics =['accuracy'])

#Fitting the classifier
classifier.fit(X_train,y_train,batch_size = 10, epochs = 100)

#predicting
y_pred = classifier.predict(X_test )

#Decoding y_pred
y_pred = np.argmax(y_pred,axis = 1)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy = cm.diagonal().sum()/cm.sum()

print("The Accuracy of the ANN in predicting the MNIST handwritten digits is {} %".format(accuracy*100 ))
