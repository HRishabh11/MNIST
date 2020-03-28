#MNIST DIGIT CLASSIFICATION USING CNN

#importig packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import  Flatten
from sklearn.metrics import confusion_matrix, classification_report

def load_data():
    #Importig the data
    from keras.datasets import mnist
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    #Normalizing
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train/255.0
    X_test = X_test/255.0
    #Reshaping the features
    X_train = X_train.reshape(X_train.shape[0],28,28,1)
    X_test = X_test.reshape(X_test.shape[0],28,28,1)
    return X_train, y_train, X_test, y_test

def model_gen():
    #Making the classifier 
    classifier = Sequential()
    classifier.add(Convolution2D(16,3,3, input_shape = (28,28,1),
                                 activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 10, activation = 'softmax'))
    
    #Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    
    return classifier

def evaluate_model(classifier,train_features, train_labels):
    #fitting the classifier
    train_label_enc = keras.utils.to_categorical(train_labels,num_classes = 10)
    classifier.fit(train_features,train_label_enc,batch_size = 32, epochs = 10)
    pred_values = classifier.predict(train_features)
    #Decoding y_pred
    pred_values = np.argmax(pred_values,axis = 1)
    cm = confusion_matrix(train_labels,pred_values)
    cr = classification_report(train_labels,pred_values)
    print("---------------------------------------------")
    print("CONFUSION MATRIX ~Training")
    print(cm)
    print("---------------------------------------------")
    print("---------------------------------------------")
    print("CLASSIFICATION REPORT ~Training")
    print(cr)
    print("---------------------------------------------")
    return classifier, pred_values

def test_accuracy(classifier, test_features, test_labels):
    pred_values = classifier.predict(test_features)
    #Decoding y_pred
    pred_values = np.argmax(pred_values,axis = 1)
    cm = confusion_matrix(test_labels,pred_values)
    cr = classification_report(test_labels,pred_values)
    print("---------------------------------------------")
    print("CONFUSION MATRIX ~Test")
    print(cm)
    print("---------------------------------------------")
    print("---------------------------------------------")
    print("CLASSIFICATION REPORT ~Test")
    print(cr)
    print("---------------------------------------------")
    return classifier, pred_values
    
